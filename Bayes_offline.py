# offline_pi_to_bayes.py
# Offline integration: Pi CSV -> yaw integration -> (pseudo)step events -> Bayes map correction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===================== USER SETTINGS =====================
CSV_PATH = "sensor_log_3.csv"   # <-- zet hier je Pi log bestand neer
USE_DT_COLUMN_IF_EXISTS = True

# columns (script probeert ook automatisch te vinden)
COL_TIME_CANDIDATES = ["timestamp", "time"]
COL_DT_CANDIDATES = ["dt"]
COL_GYROZ_CANDIDATES = ["gyro_z", "gz", "gyro_z_rad_s"]

# Movement segmentation (tuned for your "move a bit -> wait 3-10s" pattern)
# We detect MOVING when |gyro_z| exceeds threshold or accel magnitude exceeds threshold
GYRO_MOVING_THRESH = 0.08      # rad/s (laag houden; bureau-beweging geeft kleine gyro)
MIN_STILL_SECONDS = 2.0        # still duration to count as a "pause"
MIN_MOVE_SECONDS = 0.7         # move duration to count as a "move segment"

# Bayes model parameters
MAP_H, MAP_W = 100, 100        # resolution of your floorplan prior map
SIGMA_MAP = 3.0                # gaussian blur for floorplan prior (slides-style)
DRAW_VALUE = 20.0

STRIDE_PX = 8                  # how far a "move segment" moves you (pixels). Tune later.
SIGMA_R_PX = 3.0               # uncertainty of stride ring (pixels)

N_ANGLES = 360
R_MIN = 1
R_MAX_EXTRA = 6                # ray length = STRIDE_PX + R_MAX_EXTRA
SIGMA_THETA = 0.35             # rad (~20 degrees) for IMU yaw likelihood

# ===================== FLOORPLAN PRIOR (slides-like) =====================
def make_demo_floorplan_prior(h, w, draw_value=20.0, sigma=3.0):
    """
    Demo map: a big square corridor. Replace with your real floorplan later.
    """
    m = np.zeros((h, w), dtype=float)

    # big square border corridor
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

def posterior_mean_and_map(post):
    h, w = post.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total = post.sum()
    x_mean = float((post * xx).sum() / total)
    y_mean = float((post * yy).sum() / total)
    idx = int(np.argmax(post))
    y_map, x_map = np.unravel_index(idx, post.shape)
    return (x_mean, y_mean), (int(x_map), int(y_map))

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

# ===================== CSV HELPERS =====================
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_dt(df):
    dt_col = find_col(df, COL_DT_CANDIDATES)
    t_col = find_col(df, COL_TIME_CANDIDATES)

    if USE_DT_COLUMN_IF_EXISTS and dt_col is not None:
        dt = df[dt_col].astype(float).values
        # sometimes first dt is huge; clip a bit
        dt = np.clip(dt, 1e-4, 0.2)
        return dt

    if t_col is None:
        raise ValueError("No dt column and no timestamp column found.")
    t = df[t_col].astype(float).values
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.median(dt[1:50]) if len(dt) > 50 else (dt[1] if len(dt) > 1 else 0.05)
    dt = np.clip(dt, 1e-4, 0.2)
    return dt

def integrate_yaw(df, dt):
    gz_col = find_col(df, COL_GYROZ_CANDIDATES)
    if gz_col is None:
        raise ValueError(f"Could not find gyro_z column. Tried: {COL_GYROZ_CANDIDATES}")
    gz = df[gz_col].astype(float).values
    yaw = np.zeros_like(gz, dtype=float)
    for i in range(1, len(gz)):
        yaw[i] = wrap_pi(yaw[i-1] + gz[i] * dt[i])
    return gz, yaw

# ===================== SEGMENTATION: move/pause events =====================
def detect_move_events(gz, dt, gyro_thresh=0.08, min_still_s=2.0, min_move_s=0.7):
    """
    Returns list of events: (start_idx, end_idx) for MOVE segments.
    Uses |gyro_z| threshold as motion proxy.
    """
    moving = np.abs(gz) > gyro_thresh

    # Smooth a little: convert to runs of moving/still
    events = []
    i = 0
    n = len(moving)

    def run_length(mask, start):
        j = start
        while j < n and moving[j] == mask:
            j += 1
        return j

    # Find MOVE runs that are separated by sufficiently long STILL
    while i < n:
        j = run_length(moving[i], i)
        i = j

    # Build runs explicitly
    runs = []
    i = 0
    while i < n:
        mask = moving[i]
        j = i
        s = 0.0
        while j < n and moving[j] == mask:
            s += dt[j]
            j += 1
        runs.append((mask, i, j-1, s))
        i = j

    # Use still duration to define move events (your protocol: move then pause)
    for r_idx, (mask, a, b, dur) in enumerate(runs):
        if mask:  # MOVE
            if dur >= min_move_s:
                # check if previous run is still long enough OR next run is still long enough
                prev_ok = (r_idx > 0 and runs[r_idx-1][0] is False and runs[r_idx-1][3] >= min_still_s)
                next_ok = (r_idx < len(runs)-1 and runs[r_idx+1][0] is False and runs[r_idx+1][3] >= min_still_s)
                if prev_ok or next_ok:
                    events.append((a, b))
    return events

# ===================== MAIN OFFLINE PIPELINE =====================
def main():
    df = pd.read_csv(CSV_PATH)
    dt = compute_dt(df)
    gz, yaw_imu = integrate_yaw(df, dt)

    move_events = detect_move_events(gz, dt,
                                     gyro_thresh=GYRO_MOVING_THRESH,
                                     min_still_s=MIN_STILL_SECONDS,
                                     min_move_s=MIN_MOVE_SECONDS)

    print(f"[INFO] Loaded {len(df)} samples from {CSV_PATH}")
    print(f"[INFO] Found {len(move_events)} MOVE events (based on gyro_z threshold).")
    if len(move_events) == 0:
        print("[WARN] No move events detected. Lower GYRO_MOVING_THRESH or check gyro_z column.")
        return

    # Floorplan prior (demo for now)
    map_pdf = make_demo_floorplan_prior(MAP_H, MAP_W, DRAW_VALUE, SIGMA_MAP)

    # Start position (choose somewhere on corridor)
    xk, yk = int(0.2*MAP_W), int(0.2*MAP_H)
    traj_corr = [(xk, yk)]
    traj_imu  = [(xk, yk)]
    yaw_corr_list = [0.0]
    yaw_imu_list = [0.0]

    theta_prev = 0.0

    # For each move event, take yaw_imu at event end as "measurement"
    for e_idx, (a, b) in enumerate(move_events, start=1):
        theta_imu = float(yaw_imu[b])  # yaw at end of move
        # IMU-only position update (no map correction)
        x_imu, y_imu = traj_imu[-1]
        x_imu2 = int(round(x_imu + STRIDE_PX * np.cos(theta_imu)))
        y_imu2 = int(round(y_imu + STRIDE_PX * np.sin(theta_imu)))
        traj_imu.append((x_imu2, y_imu2))
        yaw_imu_list.append(theta_imu)

        # Map-corrected: position posterior -> angle prior -> fuse with IMU yaw
        prior_xy = gaussian_ring_prior(map_pdf.shape, (xk, yk), STRIDE_PX, SIGMA_R_PX)
        post_xy, ok_xy = bayes_position_update(map_pdf, prior_xy)

        angles, p_th_map = angle_prior_from_posterior(
            post_xy, center_xy=(xk, yk), n_angles=N_ANGLES,
            r_min=R_MIN, r_max=STRIDE_PX + R_MAX_EXTRA
        )
        p_th_imu = gaussian_angle_likelihood(angles, theta_imu, sigma_theta=SIGMA_THETA)
        p_th_post, ok_th = fuse_angle(p_th_map, p_th_imu)

        theta_k = float(angles[int(np.argmax(p_th_post))])

        # update corrected position using corrected theta
        xk2 = int(round(xk + STRIDE_PX * np.cos(theta_k)))
        yk2 = int(round(yk + STRIDE_PX * np.sin(theta_k)))

        print(f"[ev={e_idx:02d}] ok_xy={ok_xy} ok_th={ok_th} "
              f"theta_imu={theta_imu:+.2f} -> theta_corr={theta_k:+.2f} | "
              f"pos_corr=({xk},{yk})->({xk2},{yk2})")

        xk, yk = xk2, yk2
        traj_corr.append((xk, yk))
        yaw_corr_list.append(theta_k)
        theta_prev = theta_k

    # ===== Plot results =====
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

    plt.figure(figsize=(8,3))
    plt.title("Yaw: IMU vs corrected (per move event)")
    plt.plot(yaw_imu_list, marker="o", label="IMU yaw")
    plt.plot(yaw_corr_list, marker="o", label="Corrected yaw")
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.xlabel("event #"); plt.ylabel("rad")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
