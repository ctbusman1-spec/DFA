# Bayes_offline.py
# Robust offline: Pi CSV -> detect STILL blocks -> MOVE events between them -> map-corrected heading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===================== USER SETTINGS =====================
CSV_PATH = "sensor_log_5.csv"   # <-- zet hier jouw bestandnaam

# --- STILL detection via rolling statistics (robust) ---
WIN_SAMPLES = 20                 # ~1s if ~20Hz; ~0.4s if ~50Hz
ACC_STD_STILL_THRESH = 0.06      # m/s^2  (low variation = still)
GZ_MEAN_STILL_THRESH = 0.05      # rad/s  (low rotation = still)

MIN_STILL_SECONDS = 1.5          # pause must be at least this long
MIN_MOVE_SECONDS  = 0.4          # move segment must be at least this long

# --- Bayes model parameters ---
MAP_H, MAP_W = 100, 100
SIGMA_MAP = 3.0
DRAW_VALUE = 20.0

STRIDE_PX = 8
SIGMA_R_PX = 3.0

N_ANGLES = 360
R_MIN = 1
R_MAX_EXTRA = 6
SIGMA_THETA = 0.35               # rad (~20 deg)

# ===================== FLOORPLAN PRIOR (demo) =====================
def make_demo_floorplan_prior(h, w, draw_value=20.0, sigma=3.0):
    m = np.zeros((h, w), dtype=float)

    top, bottom = int(0.15*h), int(0.85*h)
    left, right = int(0.15*w), int(0.85*w)

    m[top, left:right] = draw_value
    m[bottom, left:right] = draw_value
    m[top:bottom, left] = draw_value
    m[top:bottom, right] = draw_value

    pdf = gaussian_filter(m, sigma=sigma)
    if np.max(pdf) > 0:
        pdf = pdf / np.max(pdf)
    pdf = pdf / np.sum(pdf)
    return pdf

# ===================== BAYES CORE =====================
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

def angle_prior_from_posterior(posterior_pdf, center_xy, n_angles=360, r_min=1, r_max=12):
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
    if total <= 0:
        p_theta[:] = 1.0 / n_angles
    else:
        p_theta /= total
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

# ===================== IMU HELPERS (match your logs) =====================
def compute_dt(df):
    dt = df["dt"].astype(float).values
    dt = np.clip(dt, 1e-4, 0.2)
    # occasional spikes exist; clip helps
    return dt

def integrate_yaw(df, dt):
    gz = df["gyro_z"].astype(float).values
    yaw = np.zeros_like(gz, dtype=float)
    for i in range(1, len(gz)):
        yaw[i] = wrap_pi(yaw[i-1] + gz[i] * dt[i])
    return gz, yaw

def accel_lin_magnitude(df):
    ax = df["accel_lin_x_mps2"].astype(float).values
    ay = df["accel_lin_y_mps2"].astype(float).values
    az = df["accel_lin_z_mps2"].astype(float).values
    return np.sqrt(ax**2 + ay**2 + az**2)

# ===================== RUN / SEGMENT HELPERS =====================
def build_runs(mask, dt):
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        m = bool(mask[i])
        j = i
        dur = 0.0
        while j < n and bool(mask[j]) == m:
            dur += float(dt[j])
            j += 1
        runs.append((m, i, j-1, dur))
        i = j
    return runs

def rolling_std(x, win):
    s = pd.Series(x).rolling(win, center=True, min_periods=win).std()
    s = s.bfill().ffill()
    return s.values

def rolling_mean(x, win):
    s = pd.Series(x).rolling(win, center=True, min_periods=win).mean()
    s = s.bfill().ffill()
    return s.values

def detect_move_events_from_still(df, dt, win=20,
                                 acc_std_thresh=0.06,
                                 gz_mean_thresh=0.05,
                                 min_still_s=1.5,
                                 min_move_s=0.4):
    """
    1) Detect STILL blocks robustly:
       - low rolling std of accel_lin magnitude
       - low rolling mean of |gyro_z|
    2) Create MOVE events as segments BETWEEN consecutive STILL blocks.
    """
    acc_mag = accel_lin_magnitude(df)
    gz_abs = np.abs(df["gyro_z"].astype(float).values)

    acc_std = rolling_std(acc_mag, win)
    gz_mean = rolling_mean(gz_abs, win)

    still_mask = (acc_std < acc_std_thresh) & (gz_mean < gz_mean_thresh)

    runs = build_runs(still_mask, dt)
    still_blocks = [(a, b, dur) for (m, a, b, dur) in runs if m and dur >= min_still_s]

    # Diagnostics
    print(f"[INFO] acc_std p10/p50/p90/max: "
          f"{np.percentile(acc_std,[10,50,90]).round(3)} / {np.max(acc_std):.3f}")
    print(f"[INFO] gz_mean p10/p50/p90/max: "
          f"{np.percentile(gz_mean,[10,50,90]).round(3)} / {np.max(gz_mean):.3f}")
    print(f"[INFO] Found {len(still_blocks)} STILL blocks (>= {min_still_s}s).")

    if len(still_blocks) < 2:
        return [], still_mask

    # MOVE events = between end of still_i and start of still_{i+1}
    events = []
    for i in range(len(still_blocks) - 1):
        a_end = still_blocks[i][1]
        b_start = still_blocks[i+1][0]
        move_start = a_end + 1
        move_end = b_start - 1
        if move_end <= move_start:
            continue
        move_dur = float(dt[move_start:move_end+1].sum())
        if move_dur >= min_move_s:
            events.append((move_start, move_end))

    return events, still_mask

# ===================== MAIN =====================
def main():
    df = pd.read_csv(CSV_PATH)

    needed = ["dt","gyro_z","accel_lin_x_mps2","accel_lin_y_mps2","accel_lin_z_mps2"]
    if not all(c in df.columns for c in needed):
        raise ValueError(f"CSV mist kolommen. Nodig: {needed}\nGevonden: {list(df.columns)}")

    dt = compute_dt(df)
    gz, yaw_imu = integrate_yaw(df, dt)

    print(f"[INFO] Loaded {len(df)} samples from {CSV_PATH}")

    move_events, still_mask = detect_move_events_from_still(
        df, dt,
        win=WIN_SAMPLES,
        acc_std_thresh=ACC_STD_STILL_THRESH,
        gz_mean_thresh=GZ_MEAN_STILL_THRESH,
        min_still_s=MIN_STILL_SECONDS,
        min_move_s=MIN_MOVE_SECONDS
    )

    print(f"[INFO] Found {len(move_events)} MOVE events (between STILL blocks).")
    if len(move_events) == 0:
        print("[WARN] No move events detected.")
        print("Try these tweaks:")
        print(" - Lower MIN_STILL_SECONDS to 1.0")
        print(" - Increase ACC_STD_STILL_THRESH to 0.08")
        print(" - Increase GZ_MEAN_STILL_THRESH to 0.08")
        return

    # Floorplan prior (demo)
    map_pdf = make_demo_floorplan_prior(MAP_H, MAP_W, DRAW_VALUE, SIGMA_MAP)

    # Start position
    xk, yk = int(0.2*MAP_W), int(0.2*MAP_H)

    traj_corr = [(xk, yk)]
    traj_imu  = [(xk, yk)]
    yaw_corr_list = [0.0]
    yaw_imu_list  = [0.0]

    for e_idx, (a, b) in enumerate(move_events, start=1):
        theta_imu = float(yaw_imu[b])

        # IMU-only step
        x_i, y_i = traj_imu[-1]
        x_i2 = int(round(x_i + STRIDE_PX * np.cos(theta_imu)))
        y_i2 = int(round(y_i + STRIDE_PX * np.sin(theta_imu)))
        traj_imu.append((x_i2, y_i2))
        yaw_imu_list.append(theta_imu)

        # Map-corrected step
        prior_xy = gaussian_ring_prior(map_pdf.shape, (xk, yk), STRIDE_PX, SIGMA_R_PX)
        post_xy, ok_xy = bayes_position_update(map_pdf, prior_xy)

        angles, p_th_map = angle_prior_from_posterior(
            post_xy, center_xy=(xk, yk), n_angles=N_ANGLES,
            r_min=R_MIN, r_max=STRIDE_PX + R_MAX_EXTRA
        )
        p_th_imu = gaussian_angle_likelihood(angles, theta_imu, sigma_theta=SIGMA_THETA)
        p_th_post, ok_th = fuse_angle(p_th_map, p_th_imu)

        theta_k = float(angles[int(np.argmax(p_th_post))])

        xk2 = int(round(xk + STRIDE_PX * np.cos(theta_k)))
        yk2 = int(round(yk + STRIDE_PX * np.sin(theta_k)))

        # keep inside bounds (just for nicer plots)
        xk2 = int(np.clip(xk2, 0, MAP_W-1))
        yk2 = int(np.clip(yk2, 0, MAP_H-1))

        print(f"[ev={e_idx:02d}] ok_xy={ok_xy} ok_th={ok_th} "
              f"theta_imu={theta_imu:+.2f} -> theta_corr={theta_k:+.2f} "
              f"| pos=({xk},{yk})->({xk2},{yk2})")

        xk, yk = xk2, yk2
        traj_corr.append((xk, yk))
        yaw_corr_list.append(theta_k)

    # ===== Plot trajectory =====
    plt.figure(figsize=(6,6))
    plt.title("Trajectory: IMU-only vs map-corrected (offline)")
    plt.imshow(map_pdf, origin="lower")

    xs_c = [p[0] for p in traj_corr]; ys_c = [p[1] for p in traj_corr]
    xs_i = [p[0] for p in traj_imu];  ys_i = [p[1] for p in traj_imu]

    plt.plot(xs_i, ys_i, marker="o", label="IMU-only")
    plt.plot(xs_c, ys_c, marker="o", label="Map-corrected")
    plt.scatter([xs_c[0]],[ys_c[0]], marker="s", s=80, label="start")
    plt.legend()
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    # ===== Plot yaw =====
    plt.figure(figsize=(8,3))
    plt.title("Yaw: IMU vs corrected (per MOVE event)")
    plt.plot(yaw_imu_list, marker="o", label="IMU yaw")
    plt.plot(yaw_corr_list, marker="o", label="Corrected yaw")
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.xlabel("event #"); plt.ylabel("rad")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
