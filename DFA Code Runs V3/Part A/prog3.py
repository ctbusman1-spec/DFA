import time
import json
from collections import deque
import paho.mqtt.client as mqtt

# --- CONFIG ---
MQTT_BROKER = "192.168.137.150"
MQTT_PORT = 1883
MQTT_TOPIC = "performance/cpu"

WINDOW_SECONDS = 1.0
PRINT_EVERY_SECONDS = 0.5

# t_received, cpu_pct)
window = deque()

def prune(now: float):
    """Remove items older than WINDOW_SECONDS from the left side."""
    cutoff = now - WINDOW_SECONDS
    while window and window[0][0] < cutoff:
        window.popleft()

def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"Connected. Subscribing to '{MQTT_TOPIC}' | window={WINDOW_SECONDS}s")
        client.subscribe(MQTT_TOPIC, qos=0)
    else:
        print(f"Connect failed: {reason_code}")

def on_message(client, userdata, msg):
    now = time.time()
    try:
        payload = json.loads(msg.payload.decode("utf-8"))

        # Program 1 publishes cpu_usage_pct
        cpu_val = float(payload.get("cpu_usage_pct", 0.0))

        window.append((now, cpu_val))
        prune(now)

    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    last_print = 0.0

    try:
        while True:
            now = time.time()
            prune(now)

            if now - last_print >= PRINT_EVERY_SECONDS:
                last_print = now

                if not window:
                    print("… waiting for data")
                else:
                    avg_cpu = sum(v for _, v in window) / len(window)
                    print(f"[Exact | window={WINDOW_SECONDS:>4.1f}s | n={len(window):>4}] avg CPU = {avg_cpu:6.2f}%")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
