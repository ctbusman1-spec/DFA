#!/usr/bin/env python3
import time
import json
import socket
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from sense_hat import SenseHat

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC = "imu/raw"

DT = 0.02  # 20 ms (50 Hz)
KEEPALIVE = 60
LOG_TO_CSV = True
CSV_PATH = "/home/groupx/imu_raw.csv"


def iso_utc_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def init_imu_with_retry(max_tries=30, sleep_s=0.2):
    s = SenseHat()
    last_err = None
    for _ in range(max_tries):
        try:
            _ = s.get_gyroscope_raw()
            _ = s.get_accelerometer_raw()
            return s
        except OSError as e:
            last_err = e
            time.sleep(sleep_s)
    raise OSError(f"IMU init failed after {max_tries} tries: {last_err}")


def safe_read(sense: SenseHat, max_tries=5, sleep_s=0.02):
    last_err = None
    for _ in range(max_tries):
        try:
            acc = sense.get_accelerometer_raw()
            gyro = sense.get_gyroscope_raw()
            return acc, gyro
        except OSError as e:
            last_err = e
            time.sleep(sleep_s)
    raise OSError(f"IMU read failed after retries: {last_err}")


def main():
    host = socket.gethostname()

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER, MQTT_PORT, KEEPALIVE)
    client.loop_start()

    # Init IMU robustly
    sense = init_imu_with_retry()
    print(f"IMU streamer started @ {1/DT:.1f} Hz. Publishing to '{TOPIC}'.")

    csv_f = None
    if LOG_TO_CSV:
        csv_f = open(CSV_PATH, "w", buffering=1)
        csv_f.write("timestamp_ms,timestamp_iso_utc,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps\n")
        print(f"Logging to: {CSV_PATH}")

    # Warm-up: read a few samples before publishing
    for _ in range(10):
        try:
            _ = sense.get_accelerometer_raw()
            _ = sense.get_gyroscope_raw()
        except OSError:
            pass
        time.sleep(0.05)

    try:
        next_t = time.perf_counter()
        zero_accel_streak = 0

        while True:
            next_t += DT

            ts_ms = int(time.time() * 1000)
            ts_iso = iso_utc_ms()

            acc, gyro = safe_read(sense)

            ax, ay, az = float(acc["x"]), float(acc["y"]), float(acc["z"])
            gx, gy, gz = float(gyro["x"]), float(gyro["y"]), float(gyro["z"])

            # Detect suspicious accel=0,0,0 streak
            if ax == 0 and ay == 0 and az == 0:
                zero_accel_streak += 1
                if zero_accel_streak == 25:  # ~0.5s at 50 Hz
                    print("⚠️ Warning: accelerometer reading is (0,0,0) repeatedly. Check Sense HAT / drivers.")
            else:
                zero_accel_streak = 0

            msg = {
                "timestamp_ms": ts_ms,
                "timestamp_iso_utc": ts_iso,
                "host": host,
                "dt_s": DT,
                "accel_g": {"x": ax, "y": ay, "z": az},
                "gyro_dps": {"x": gx, "y": gy, "z": gz},
            }

            client.publish(TOPIC, json.dumps(msg), qos=0)

            if csv_f:
                csv_f.write(f"{ts_ms},{ts_iso},{ax},{ay},{az},{gx},{gy},{gz}\n")

            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if csv_f:
            csv_f.close()
        client.loop_stop()
        client.disconnect()
        print("MQTT disconnected.")


if __name__ == "__main__":
    main()
