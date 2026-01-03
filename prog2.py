#!/usr/bin/env python3
"""
DFA - Program 2
Publishes Sense HAT data and a Bayesian predicted location estimate.

- Reads Sense HAT accelerometer every ~10 ms (if possible)
- Uses a simple 2D Kalman filter (Bayesian estimator) with acceleration as input
- When stationary, applies a zero-velocity update to reduce drift
- Publishes:
    * raw sensor stream:        sensehat/raw_data
    * predicted location stream: sensehat/predicted_location
"""

import json
import time
import math
import socket
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt
from sense_hat import SenseHat

# === CONFIG ===
MQTT_BROKER = "localhost"          # Mosquitto on the Pi
MQTT_PORT = 1883
RAW_TOPIC = "sensehat/raw_data"
PRED_TOPIC = "sensehat/predicted_location"

DT = 0.01                          # 10 ms
QOS = 0
KEEPALIVE = 60

# Stationary detection threshold (m/s^2) for |a_xy|
# If device is lying still, a_x and a_y should be near 0.
STATIONARY_ACCEL_THRESH = 0.2

# How often to print a short status line (seconds)
PRINT_EVERY_S = 1.0


def iso_utc_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def main():
    sense = SenseHat()
    host = socket.gethostname()

    # MQTT client
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER, MQTT_PORT, KEEPALIVE)
    client.loop_start()

    # ----------------------------
    # Bayesian estimator: Kalman filter
    # State: [x, y, vx, vy]^T  (meters, meters, m/s, m/s)
    # Control input: ax, ay (m/s^2) from accelerometer
    # Process model:
    #   x_{k+1} = x + vx*dt + 0.5*ax*dt^2
    #   y_{k+1} = y + vy*dt + 0.5*ay*dt^2
    #   vx_{k+1}= vx + ax*dt
    #   vy_{k+1}= vy + ay*dt
    # Measurement update (only when stationary): z = [vx, vy] ≈ [0, 0]
    # ----------------------------

    x = np.zeros((4, 1))  # initial at origin, zero velocity

    F = np.array([
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)

    B = np.array([
        [0.5 * DT * DT, 0],
        [0, 0.5 * DT * DT],
        [DT, 0],
        [0, DT],
    ], dtype=float)

    # Process noise (tuneable). Larger => more uncertainty growth.
    q = 0.2
    Q = (q ** 2) * np.eye(4)

    # Measurement model for zero-velocity update (observe vx, vy)
    H = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    # Measurement noise for ZUPT (small => strong pull to zero)
    r = 0.05
    R = (r ** 2) * np.eye(2)

    P = 0.1 * np.eye(4)  # initial covariance

    # Prime timing
    last_print = 0.0

    print("Program 2 started: Sense HAT → Bayesian predicted location (Kalman).")
    print(f"Publishing RAW to '{RAW_TOPIC}' and PRED to '{PRED_TOPIC}' every ~{int(DT*1000)} ms.")

    try:
        while True:
            t0 = time.perf_counter()

            # --- Read Sense HAT accelerometer (in g) ---
            acc = sense.get_accelerometer_raw()
            ax = float(acc["x"]) * 9.81  # m/s^2
            ay = float(acc["y"]) * 9.81  # m/s^2
            az = float(acc["z"]) * 9.81  # m/s^2

            # Basic stationary detection (planar)
            a_xy = math.sqrt(ax * ax + ay * ay)
            stationary = a_xy < STATIONARY_ACCEL_THRESH

            # Timestamps
            ts_ms = int(time.time() * 1000)
            ts_iso = iso_utc_ms()

            # --- Publish raw Sense HAT data ---
            raw_msg = {
                "timestamp_ms": ts_ms,
                "timestamp_iso_utc": ts_iso,
                "host": host,
                "accel_mps2": {"x": ax, "y": ay, "z": az},
                "a_xy": a_xy,
                "stationary": stationary,
            }
            client.publish(RAW_TOPIC, json.dumps(raw_msg), qos=QOS)

            # --- Kalman predict step (Bayesian prediction) ---
            u = np.array([[ax], [ay]], dtype=float)
            x = F @ x + B @ u
            P = F @ P @ F.T + Q

            # --- Optional measurement update (Bayesian correction) ---
            # If device is stationary, enforce vx, vy ≈ 0 (ZUPT)
            if stationary:
                z = np.array([[0.0], [0.0]])
                y_res = z - (H @ x)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y_res
                P = (np.eye(4) - K @ H) @ P

            # --- Publish predicted location ---
            pred_msg = {
                "timestamp_ms": ts_ms,
                "timestamp_iso_utc": ts_iso,
                "host": host,
                "status": "predicted",
                "bayesian_model": "kalman_2d_cv_with_zupt",
                "dt_s": DT,
                "position_m": {"x": float(x[0, 0]), "y": float(x[1, 0])},
                "velocity_mps": {"x": float(x[2, 0]), "y": float(x[3, 0])},
            }
            client.publish(PRED_TOPIC, json.dumps(pred_msg), qos=QOS)

            # Light logging
            now = time.time()
            if now - last_print >= PRINT_EVERY_S:
                last_print = now
                print(
                    f"[{ts_iso}] pos=({pred_msg['position_m']['x']:.3f}, {pred_msg['position_m']['y']:.3f}) m | "
                    f"vel=({pred_msg['velocity_mps']['x']:.3f}, {pred_msg['velocity_mps']['y']:.3f}) m/s | "
                    f"stationary={stationary}"
                )

            # Keep ~10ms cycle time
            elapsed = time.perf_counter() - t0
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("MQTT disconnected.")


if __name__ == "__main__":
    main()
