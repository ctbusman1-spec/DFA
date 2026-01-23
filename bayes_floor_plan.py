# bayes_floor_plan.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===================== SETTINGS =====================
MAP_H, MAP_W = 20, 20
SIGMA_MAP = 0.8          # map smoothing (slides: 0.5; iets hoger is vaak mooier)
DRAW_VALUE = 10.0

START_XY = (5, 5)        # (x,y)
N_STEPS = 12

STRIDE_PX = 4
SIGMA_R_PX = 1.5

PICK_MODE = "MEAN"       # "MEAN" voorkomt ping-pong; probeer ook "MAP"

# ===================== FLOORPLAN PRIOR (slides style) =====================
def make_square_corridor_prior(h, w, draw_value=10.0, sigma=0.8):
    floor_map = np.zeros((h, w), dtype=float)

    # Square border corridor (your example)
    floor_map[5, 5:15] = draw_value
    floor_map[14, 5:15] = draw_value
    floor_map[5:15, 5] = draw_value
    floor_map[5:15, 14] = draw_value

    pdf_map = gaussian_filter(floor_map, sigma=sigma)

    # Normalize to max=1 (slides)
    m = np.max(pdf_map)
    if m > 0:
        pdf_map = pdf_map / m

    # Also make it sum to 1 (true PDF). This is nicer for Bayes.
    pdf_map = pdf_map / np.sum(pdf_map)
    return pdf_map

# ===================== PRIORS & UPDATE =====================
def gaussian_ring_prior(shape, center_xy, radius_px, sigma_r_px):
    h, w = shape
    cx, cy = center_xy
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    prior = np.exp(-0.5 * ((dist - radius_px) / sigma_r_px) ** 2)
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

def pick_next(mean_xy, map_xy, mode="MEAN"):
    if mode.upper() == "MAP":
        return map_xy
    return (int(round(mean_xy[0])), int(round(mean_xy[1])))


def angle_prior_from_posterior(posterior_pdf, center_xy, n_angles=360, r_min=1, r_max=8):
    """
    Projecteer 2D posterior p(x,y) naar een orientation prior p(theta).
    Voor elke theta: som van p langs een straal vanuit center_xy.
    Output: angles (rad), p_theta (genormaliseerd).
    """
    h, w = posterior_pdf.shape
    cx, cy = center_xy

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    p_theta = np.zeros(n_angles, dtype=float)

    for i, th in enumerate(angles):
        # sample points along the ray
        s = 0.0
        for r in range(r_min, r_max + 1):
            x = int(round(cx + r * np.cos(th)))
            y = int(round(cy + r * np.sin(th)))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            s += posterior_pdf[y, x]
        p_theta[i] = s

    # normalize to sum=1
    total = p_theta.sum()
    if total <= 0:
        # fallback uniform
        p_theta[:] = 1.0 / n_angles
    else:
        p_theta /= total

    return angles, p_theta


def gaussian_angle_likelihood(angles, theta_imu, sigma_theta=0.35):
    """
    IMU yaw likelihood p(theta | imu) as wrapped Gaussian on circle.
    sigma_theta in radians (~0.35 rad = 20 degrees).
    """
    # wrap difference to [-pi, pi]
    d = (angles - theta_imu + np.pi) % (2.0 * np.pi) - np.pi
    w = np.exp(-0.5 * (d / sigma_theta) ** 2)
    w /= w.sum()
    return w


def fuse_angle_prior_with_imu(p_theta_map, p_theta_imu, eps=1e-12):
    """
    Bayes fuse: p(theta) ∝ p(theta|map) * p(theta|imu)
    """
    post = p_theta_map * p_theta_imu
    s = post.sum()
    if s < eps:
        post = p_theta_map.copy()
        post /= post.sum()
        return post, False
    return post / s, True


def pick_angle_map(angles, p_theta):
    """Return theta (rad) at MAP."""
    return float(angles[int(np.argmax(p_theta))])


def step_position_update(xy, stride_px, theta):
    """x_k = x_{k-1} + d cos(theta), y_k = y_{k-1} + d sin(theta)"""
    x, y = xy
    x2 = int(round(x + stride_px * np.cos(theta)))
    y2 = int(round(y + stride_px * np.sin(theta)))
    return (x2, y2)



# ===================== MAIN =====================
def main():
    map_pdf = make_square_corridor_prior(MAP_H, MAP_W, DRAW_VALUE, SIGMA_MAP)

    xk, yk = START_XY
    traj = [(xk, yk)]

    print(f"[INFO] Map size: {MAP_W}x{MAP_H}, sigma_map={SIGMA_MAP}")
    print(f"[INFO] Start: {START_XY}, stride={STRIDE_PX}, sigma_r={SIGMA_R_PX}, pick={PICK_MODE}\n")

    # --- init heading (demo) ---
    theta_prev = 0.0  # start heading (rad). Zet bv. 0 = +x richting

    for k in range(1, N_STEPS + 1):
        # 1) Position update (same as before)
        prior = gaussian_ring_prior(map_pdf.shape, (xk, yk), STRIDE_PX, SIGMA_R_PX)
        post_xy, ok_xy = bayes_position_update(map_pdf, prior)

        # 2) Orientation prior from map posterior
        angles, p_theta_map = angle_prior_from_posterior(
            post_xy, center_xy=(xk, yk), n_angles=360, r_min=1, r_max=STRIDE_PX + 3
        )

        # 3) IMU yaw likelihood (DEMO)
        # In the real system theta_imu comes from gyro integration / your IMU pipeline.
        # Here we simulate it as "keep going roughly forward" + noise.
        theta_imu = theta_prev + np.random.normal(0.0, 0.15)  # ~8.6 deg noise
        p_theta_imu = gaussian_angle_likelihood(angles, theta_imu, sigma_theta=0.35)

        # 4) Fuse map prior with IMU likelihood
        p_theta_post, ok_th = fuse_angle_prior_with_imu(p_theta_map, p_theta_imu)

        # 5) Pick theta (MAP) and update position using stride + heading
        theta_k = pick_angle_map(angles, p_theta_post)
        nxt = step_position_update((xk, yk), STRIDE_PX, theta_k)

        print(f"[k={k:02d}] ok_xy={ok_xy} ok_th={ok_th} "
              f"xk=({xk},{yk}) -> next={nxt} | theta_imu={theta_imu:.2f} | theta={theta_k:.2f}")

        theta_prev = theta_k
        xk, yk = nxt
        traj.append((xk, yk))

    # Plot: map + trajectory
    plt.figure(figsize=(6, 6))
    plt.title("Bayes update: stride ring prior × floorplan prior")
    plt.imshow(map_pdf, origin="lower")
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    plt.plot(xs, ys, marker="o")
    plt.scatter([traj[0][0]], [traj[0][1]], marker="s", s=100, label="start")
    plt.scatter([traj[-1][0]], [traj[-1][1]], marker="X", s=100, label="end")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    main()
