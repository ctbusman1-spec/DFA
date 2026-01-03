#!/usr/bin/env python3
"""
DFA - Program 1 (DSMS)
CPU performance publisher on Raspberry Pi:
- Collects system performance metrics using psutil
- Publishes JSON payload to MQTT at ~10ms interval (if possible)
- Includes timestamps (epoch ms + ISO-UTC)
"""

import json
import time
import socket
from datetime import datetime, timezone

import psutil
import paho.mqtt.client as mqtt

# === CONFIG ===
MQTT_BROKER = "localhost"          # Broker runs on the Pi
MQTT_PORT = 1883
MQTT_TOPIC = "performance/cpu"

PUBLISH_INTERVAL_S = 0.01          # ~10ms
QOS = 0                            # assignment doesn't require QoS>0; keep lightweight
KEEPALIVE = 60

# Print a short status line every N seconds (avoid console spam at 100 Hz)
PRINT_EVERY_S = 1.0


def iso_utc_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def get_cpu_temp_c():
    """Best-effort CPU temperature. Returns None if unavailable."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        # Raspberry Pi commonly exposes 'cpu_thermal'
        for key in temps:
            if temps[key]:
                return temps[key][0].current
    except Exception:
        return None
    return None


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("✅ MQTT connected")
    else:
        print(f"❌ MQTT connect failed, reason_code={reason_code}")


def on_disconnect(client, userdata, reason_code, properties=None):
    # 0 = clean disconnect; nonzero means unexpected drop
    if reason_code != 0:
        print(f"⚠️ MQTT disconnected unexpectedly, reason_code={reason_code}")


def main():
    hostname = socket.gethostname()

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    # Auto-reconnect behavior
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    print(f"Poging tot verbinden met MQTT broker {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, KEEPALIVE)
    client.loop_start()

    # Prime cpu_percent to avoid weird first measurement
    psutil.cpu_percent(None)

    print(f"Publishing to topic '{MQTT_TOPIC}' every ~{int(PUBLISH_INTERVAL_S*1000)} ms ... (Ctrl+C to stop)")

    last_print = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            now_ms = int(time.time() * 1000)

            cpu_freq = psutil.cpu_freq()
            cpu_stats = psutil.cpu_stats()
            vm = psutil.virtual_memory()

            payload = {
                # Timestamps (important for time alignment in DSMS)
                "timestamp_ms": now_ms,
                "timestamp_iso_utc": iso_utc_ms(),

                # Identification (useful if you later add more Pis)
                "host": hostname,

                # Performance metrics
                "cpu_usage_pct": psutil.cpu_percent(None),
                "cpu_mhz": cpu_freq.current if cpu_freq else None,
                "ram_pct": vm.percent,
                "ctx_switches": cpu_stats.ctx_switches,

                # Optional extras (nice-to-have, harmless if None)
                "cpu_temp_c": get_cpu_temp_c(),
            }

            client.publish(MQTT_TOPIC, json.dumps(payload), qos=QOS)

            # Light logging (once per second)
            now = time.time()
            if now - last_print >= PRINT_EVERY_S:
                last_print = now
                print(f"[{payload['timestamp_iso_utc']}] published cpu={payload['cpu_usage_pct']:.1f}% ram={payload['ram_pct']:.1f}%")

            # Keep ~10ms cycle time, accounting for processing time
            elapsed = time.perf_counter() - t0
            sleep_time = PUBLISH_INTERVAL_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            # If sleep_time <= 0: system is too slow to hit 100Hz; that's okay ("if possible")

    except KeyboardInterrupt:
        print("\nProgramma gestopt door gebruiker.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Verbinding verbroken.")


if __name__ == "__main__":
    main()
