#!/usr/bin/env python3
from __future__ import annotations


import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
import math
import socket
import time

import numpy as np

from dsms.common import build_client, iso_utc_ms, load_dsms_config

try:
    from sense_hat import SenseHat
except Exception:
    SenseHat = None


def main():
    if SenseHat is None:
        raise RuntimeError("sense_hat is not available on this machine.")

    parser = argparse.ArgumentParser(description="Program 2: publish Sense HAT data and predicted location.")
    parser.add_argument("--config", default="config/dsms_config.yaml")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_dsms_config(args.config)
    broker = args.broker or cfg["mqtt"]["broker"]
    port = args.port or cfg["mqtt"]["port"]
    keepalive = cfg["mqtt"]["keepalive"]
    raw_topic = cfg["topics"]["sense_raw"]
    pred_topic = cfg["topics"]["sense_predicted"]
    dt = cfg["program2"]["dt_s"]
    qos = cfg["program2"]["qos"]
    print_every_s = cfg["program2"]["print_every_s"]
    stationary_thresh = cfg["program2"]["stationary_accel_thresh_mps2"]

    sense = SenseHat()
    host = socket.gethostname()
    client = build_client()
    client.connect(broker, port, keepalive)
    client.loop_start()

    x = np.zeros((4, 1), dtype=float)
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    B = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], dtype=float)
    q = 0.2
    Q = (q ** 2) * np.eye(4)
    H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    r = 0.05
    R = (r ** 2) * np.eye(2)
    P = 0.1 * np.eye(4)
    last_print = 0.0

    print(f"Program 2 publishing RAW to {raw_topic} and prediction to {pred_topic}")

    try:
        while True:
            t0 = time.perf_counter()
            acc = sense.get_accelerometer_raw()
            ax = float(acc["x"]) * 9.81
            ay = float(acc["y"]) * 9.81
            az = float(acc["z"]) * 9.81
            a_xy = math.sqrt(ax * ax + ay * ay)
            stationary = a_xy < stationary_thresh

            ts_ms = int(time.time() * 1000)
            ts_iso = iso_utc_ms()

            raw_msg = {
                "timestamp_ms": ts_ms,
                "timestamp_iso_utc": ts_iso,
                "host": host,
                "accel_mps2": {"x": ax, "y": ay, "z": az},
                "a_xy": a_xy,
                "stationary": stationary,
            }
            client.publish(raw_topic, json.dumps(raw_msg), qos=qos)

            u = np.array([[ax], [ay]], dtype=float)
            x = F @ x + B @ u
            P = F @ P @ F.T + Q

            if stationary:
                z = np.array([[0.0], [0.0]])
                y_res = z - (H @ x)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y_res
                P = (np.eye(4) - K @ H) @ P

            pred_msg = {
                "timestamp_ms": ts_ms,
                "timestamp_iso_utc": ts_iso,
                "host": host,
                "status": "predicted",
                "bayesian_model": "kalman_2d_cv_with_zupt",
                "dt_s": dt,
                "position_m": {"x": float(x[0, 0]), "y": float(x[1, 0])},
                "velocity_mps": {"x": float(x[2, 0]), "y": float(x[3, 0])},
            }
            client.publish(pred_topic, json.dumps(pred_msg), qos=qos)

            now = time.time()
            if now - last_print >= print_every_s:
                last_print = now
                print(
                    f"[{ts_iso}] pos=({pred_msg['position_m']['x']:.3f}, {pred_msg['position_m']['y']:.3f}) m | "
                    f"vel=({pred_msg['velocity_mps']['x']:.3f}, {pred_msg['velocity_mps']['y']:.3f}) m/s | "
                    f"stationary={stationary}"
                )

            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nProgram 2 stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
