# pi_stream.py
import time, json, csv
from datetime import datetime
import numpy as np
from sense_hat import SenseHat
import paho.mqtt.client as mqtt

# ===== USER SETTINGS =====
MQTT_HOST = "192.168.137.1"   # <-- laptop IP (hotspot meestal 192.168.137.1)
MQTT_PORT = 1884
TOPIC = "dfa/imu"

RATE_HZ = 20
LOG_TO_CSV = True
LOG_NAME = "sensor_log_live.csv"   # <-- makkelijk aanpassen

# =========================
sense = SenseHat()
sense.set_imu_config(True, True, False)  # gyro+accel, mag off

client = mqtt.Client()
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.loop_start()

dt_target = 1.0 / RATE_HZ
t_prev = time.time()

csvfile = None
writer = None

if LOG_TO_CSV:
    csvfile = open(LOG_NAME, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow([
        "timestamp","dt",
        "gyro_x","gyro_y","gyro_z",
        "accel_raw_x","accel_raw_y","accel_raw_z"
    ])

print(f"[PI] Streaming IMU -> mqtt://{MQTT_HOST}:{MQTT_PORT} topic={TOPIC} @ {RATE_HZ}Hz")
print("[PI] Ctrl+C to stop.")

try:
    while True:
        t_now = time.time()
        dt = max(t_now - t_prev, 1e-4)
        t_prev = t_now

        g = sense.get_gyroscope_raw()        # rad/s
        a = sense.get_accelerometer_raw()    # in g units

        payload = {
            "t": t_now,
            "dt": dt,
            "gx": float(g["x"]), "gy": float(g["y"]), "gz": float(g["z"]),
            "ax_g": float(a["x"]), "ay_g": float(a["y"]), "az_g": float(a["z"]),
        }

        client.publish(TOPIC, json.dumps(payload), qos=0)

        if writer:
            writer.writerow([
                t_now, dt,
                payload["gx"], payload["gy"], payload["gz"],
                payload["ax_g"], payload["ay_g"], payload["az_g"]
            ])

        # rate control
        elapsed = time.time() - t_now
        sleep = dt_target - elapsed
        if sleep > 0:
            time.sleep(sleep)

except KeyboardInterrupt:
    print("\n[PI] Stopping...")
finally:
    client.loop_stop()
    client.disconnect()
    if csvfile:
        csvfile.close()
