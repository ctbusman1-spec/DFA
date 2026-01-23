# laptop_live_bayes.py
import json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import paho.mqtt.client as mqtt
from collections import deque

# ===== USER SETTINGS =====
MQTT_HOST = "0.0.0.0"   # broker draait lokaal
MQTT_PORT = 1884
TOPIC = "dfa/imu"

RATE_HINT = 20  # alleen voor window sizing

# Autocalibratie (stil starten!)
CALIB_SECONDS = 8.0

# STILL logic: thresholds worden automatisch geleerd.
WIN_SECONDS = 1.0
MIN_STILL_SECONDS = 1.2
MIN_MOVE_SECONDS = 0.4

# Bayes demo map
MAP_H, MAP_W = 100, 100
SIGMA_MAP = 3.0
DRAW_VALUE = 20.0

STRIDE_PX = 8
SIGMA_R_PX = 3.0
N_ANGLES = 360
SIGMA_THETA = 0.35

# ================== MAP / BAYES ==================
def make_demo_floorplan_prior(h, w, draw_value=20.0, sigma=3.0):
    m = np.zeros((h, w), dtype=float)
    top, bottom = int(0.15*h), int(0.85*h)
    left, right = int(0.15*w), int(0.85*w)
    m[top, left:right] = draw_value
    m[bottom, left:right] = draw_value
    m[top:bottom, left] = draw_value
    m[top:bottom, right] = draw_value
    pdf = gaussian_filter(m, sigma=sigma)
    pdf = pdf / np.max(pdf) if np.max(pdf) > 0 else pdf
    pdf = pdf / np.sum(pdf) if np.sum(pdf) > 0 else pdf
    return pdf

def gaussian_ring_prior(shape, center_xy, radius_px, sigma_r_px):
    h, w = shape
    cx, cy = center_xy
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    prior = np.exp(-0.5 * ((dist - radius_px)/sigma_r_px)**2)
    prior /= prior.sum()
    return prior

def bayes_position_update(map_pdf, prior_pdf, eps=1e-12):
    post = map_pdf * prior_pdf
    s = post.sum()
    if s < eps:
        post = prior_pdf.copy()
        post /= post.sum()
        return post, False
    return post / s, True

def angle_prior_from_posterior(posterior_pdf, center_xy, n_angles=360, r_min=1, r_max=14):
    h, w = posterior_pdf.shape
    cx, cy = center_xy
    angles = np.linspace(0.0, 2*np.pi, n_angles, endpoint=False)
    p_theta = np.zeros(n_angles, dtype=float)
    for i, th in enumerate(angles):
        s = 0.0
        for r in range(r_min, r_max+1):
            x = int(round(cx + r*np.cos(th)))
            y = int(round(cy + r*np.sin(th)))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            s += posterior_pdf[y, x]
        p_theta[i] = s
    total = p_theta.sum()
    p_theta = (np.ones(n_angles)/n_angles) if total <= 0 else (p_theta/total)
    return angles, p_theta

def gaussian_angle_likelihood(angles, theta_imu, sigma_theta=0.35):
    d = (angles - theta_imu + np.pi) % (2*np.pi) - np.pi
    w = np.exp(-0.5 * (d/sigma_theta)**2)
    w /= w.sum()
    return w

def fuse_angle(p_map, p_imu, eps=1e-12):
    post = p_map * p_imu
    s = post.sum()
    if s < eps:
        post = p_map.copy()
        post /= post.sum()
        return post, False
    return post / s, True

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# ================== LIVE STATE ==================
map_pdf = make_demo_floorplan_prior(MAP_H, MAP_W, DRAW_VALUE, SIGMA_MAP)

# start pose (op corridor)
xk, yk = int(0.2*MAP_W), int(0.2*MAP_H)
traj_imu = [(xk, yk)]
traj_corr = [(xk, yk)]

# Buffers
buf_dt = deque(maxlen=5000)
buf_gnorm = deque(maxlen=5000)
buf_adyn = deque(maxlen=5000)
buf_gz = deque(maxlen=5000)

# For event segmentation
state = "CALIB"
calib_t0 = None
calib_g = []
calib_a = []
thresholds_ready = False

# thresholds learned from calib
GNORM_STILL_T = None
ADYN_STILL_T = None

# current segment
still_time = 0.0
move_time = 0.0
in_move = False
move_gz_integral = 0.0  # yaw integrated over move segment (gyro_z)
theta_imu_total = 0.0   # cumulative yaw (optional)
last_print = time.time()

# Rolling window
WIN_SAMPLES = max(5, int(WIN_SECONDS * RATE_HINT))

# ================== PLOT SETUP ==================
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(map_pdf, origin="lower")
line_imu, = ax.plot([], [], marker="o", label="IMU-only")
line_cor, = ax.plot([], [], marker="o", label="Map-corrected")
ax.legend()
ax.set_title("Live: IMU-only vs map-corrected")

def redraw():
    xs_i = [p[0] for p in traj_imu]; ys_i = [p[1] for p in traj_imu]
    xs_c = [p[0] for p in traj_corr]; ys_c = [p[1] for p in traj_corr]
    line_imu.set_data(xs_i, ys_i)
    line_cor.set_data(xs_c, ys_c)
    ax.set_xlim(0, MAP_W-1)
    ax.set_ylim(0, MAP_H-1)
    fig.canvas.draw()
    fig.canvas.flush_events()

# ================== MQTT CALLBACK ==================
def on_message(client, userdata, msg):
    global calib_t0, thresholds_ready, GNORM_STILL_T, ADYN_STILL_T
    global still_time, move_time, in_move, move_gz_integral, theta_imu_total
    global xk, yk, last_print

    data = json.loads(msg.payload.decode("utf-8"))
    dt = float(data["dt"])
    gx, gy, gz = float(data["gx"]), float(data["gy"]), float(data["gz"])
    ax_g, ay_g, az_g = float(data["ax_g"]), float(data["ay_g"]), float(data["az_g"])

    # gyro norm (orientation-proof for motion)
    gnorm = float(np.sqrt(gx*gx + gy*gy + gz*gz))

    # accel in m/s^2
    ax = ax_g * 9.81
    ay = ay_g * 9.81
    az = az_g * 9.81

    # dynamic accel magnitude (remove gravity approximately via magnitude)
    # crude but robust for "still vs moving"
    amag = float(np.sqrt(ax*ax + ay*ay + az*az))
    adyn = abs(amag - 9.81)

    buf_dt.append(dt)
    buf_gnorm.append(gnorm)
    buf_adyn.append(adyn)
    buf_gz.append(gz)

    # --- calibration: assume user is still at start ---
    if not thresholds_ready:
        if calib_t0 is None:
            calib_t0 = time.time()
        calib_g.append(gnorm)
        calib_a.append(adyn)
        if (time.time() - calib_t0) >= CALIB_SECONDS and len(calib_g) > 20:
            g_med = float(np.median(calib_g))
            g_mad = float(np.median(np.abs(np.array(calib_g) - g_med))) + 1e-6
            a_med = float(np.median(calib_a))
            a_mad = float(np.median(np.abs(np.array(calib_a) - a_med))) + 1e-6

            # learned STILL thresholds (higher => easier to be still; lower => stricter)
            GNORM_STILL_T = g_med + 6.0 * g_mad
            ADYN_STILL_T  = a_med + 6.0 * a_mad

            thresholds_ready = True
            print(f"[CALIB] GNORM_STILL_T={GNORM_STILL_T:.4f} rad/s | ADYN_STILL_T={ADYN_STILL_T:.4f} m/s^2")
            print("[CALIB] Start walking now (with pauses).")
        return

    # rolling stats (last window)
    if len(buf_gnorm) < WIN_SAMPLES:
        return

    gwin = np.array(list(buf_gnorm)[-WIN_SAMPLES:])
    awin = np.array(list(buf_adyn)[-WIN_SAMPLES:])

    g_mean = float(np.mean(gwin))
    a_mean = float(np.mean(awin))

    is_still = (g_mean < GNORM_STILL_T) and (a_mean < ADYN_STILL_T)

    # online segmentation: still -> move -> still produces 1 "event"
    if is_still:
        still_time += dt
        if in_move:
            # we are ending a move segment
            if move_time >= MIN_MOVE_SECONDS and still_time >= MIN_STILL_SECONDS:
                # finalize MOVE event
                theta_imu_total = wrap_pi(theta_imu_total + move_gz_integral)

                # IMU-only step update
                xi, yi = traj_imu[-1]
                xi2 = int(round(xi + STRIDE_PX * np.cos(theta_imu_total)))
                yi2 = int(round(yi + STRIDE_PX * np.sin(theta_imu_total)))
                traj_imu.append((xi2, yi2))

                # Map-corrected step
                prior_xy = gaussian_ring_prior(map_pdf.shape, (xk, yk), STRIDE_PX, SIGMA_R_PX)
                post_xy, ok_xy = bayes_position_update(map_pdf, prior_xy)

                angles, p_th_map = angle_prior_from_posterior(
                    post_xy, center_xy=(xk, yk),
                    n_angles=N_ANGLES, r_min=1, r_max=STRIDE_PX + 6
                )
                p_th_imu = gaussian_angle_likelihood(angles, theta_imu_total, sigma_theta=SIGMA_THETA)
                p_th_post, ok_th = fuse_angle(p_th_map, p_th_imu)

                theta_corr = float(angles[int(np.argmax(p_th_post))])

                xk2 = int(round(xk + STRIDE_PX * np.cos(theta_corr)))
                yk2 = int(round(yk + STRIDE_PX * np.sin(theta_corr)))
                xk2 = int(np.clip(xk2, 0, MAP_W-1))
                yk2 = int(np.clip(yk2, 0, MAP_H-1))

                xk, yk = xk2, yk2
                traj_corr.append((xk, yk))

                if time.time() - last_print > 0.4:
                    print(f"[EV] move_time={move_time:.2f}s | theta_imu={theta_imu_total:+.2f} | theta_corr={theta_corr:+.2f} | ok_xy={ok_xy} ok_th={ok_th}")
                    last_print = time.time()

                redraw()

            # reset move segment tracking (we are still)
            in_move = False
            move_time = 0.0
            move_gz_integral = 0.0
        else:
            # remain still
            pass
    else:
        # moving
        move_time += dt
        still_time = 0.0
        in_move = True
        # integrate yaw during move using gz (assumes device kept roughly flat; that’s OK for assignment)
        move_gz_integral = wrap_pi(move_gz_integral + float(gz) * dt)

def main():
    print("[LAPTOP] Make sure Mosquitto broker is running on this machine (port 1883).")
    print("[LAPTOP] Start script, keep Pi still for calibration, then walk with pauses.")

    client = mqtt.Client()
    client.on_message = on_message
    client.connect("127.0.0.1", MQTT_PORT, 60)
    client.subscribe(TOPIC)
    client.loop_start()

    try:
        while True:
            plt.pause(0.05)
    except KeyboardInterrupt:
        print("\n[LAPTOP] Stopping...")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
