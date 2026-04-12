#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml


def iso_utc_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_dsms_config(path: str = "../IMU assignment/config/dsms_config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (Path(__file__).resolve().parent / cfg_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_client(on_connect=None, on_message=None, on_disconnect=None):
    import paho.mqtt.client as mqtt

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    if on_connect is not None:
        client.on_connect = on_connect
    if on_message is not None:
        client.on_message = on_message
    if on_disconnect is not None:
        client.on_disconnect = on_disconnect
    return client


def get_cpu_temp_c():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in temps:
            if temps[key]:
                return temps[key][0].current
    except Exception:
        return None
    return None


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("MQTT connected" if reason_code == 0 else f"MQTT connect failed, reason_code={reason_code}")


def on_disconnect(client, userdata, reason_code, properties=None):
    if reason_code != 0:
        print(f"MQTT disconnected unexpectedly, reason_code={reason_code}")


def main():
    parser = argparse.ArgumentParser(description="Program 1: publish CPU performance metrics.")
    parser.add_argument("--config", default="../IMU assignment/config/dsms_config.yaml")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--interval", type=float, default=None)
    args = parser.parse_args()

    cfg = load_dsms_config(args.config)
    broker = args.broker or cfg["mqtt"]["broker"]
    port = args.port or cfg["mqtt"]["port"]
    topic = args.topic or cfg["topics"]["cpu"]
    interval_s = args.interval or cfg["program1"]["publish_interval_s"]
    qos = cfg["program1"]["qos"]
    keepalive = cfg["mqtt"]["keepalive"]
    print_every_s = cfg["program1"]["print_every_s"]

    hostname = socket.gethostname()
    client = build_client(on_connect=on_connect, on_disconnect=on_disconnect)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(broker, port, keepalive)
    client.loop_start()

    psutil.cpu_percent(None)
    print(f"Publishing to {topic} every ~{int(interval_s * 1000)} ms")
    last_print = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            now_ms = int(time.time() * 1000)
            cpu_freq = psutil.cpu_freq()
            cpu_stats = psutil.cpu_stats()
            vm = psutil.virtual_memory()

            payload = {
                "timestamp_ms": now_ms,
                "timestamp_iso_utc": iso_utc_ms(),
                "host": hostname,
                "cpu_usage_pct": psutil.cpu_percent(None),
                "cpu_mhz": cpu_freq.current if cpu_freq else None,
                "ram_pct": vm.percent,
                "ctx_switches": cpu_stats.ctx_switches,
                "cpu_temp_c": get_cpu_temp_c(),
            }
            client.publish(topic, json.dumps(payload), qos=qos)

            now = time.time()
            if now - last_print >= print_every_s:
                last_print = now
                print(f"[{payload['timestamp_iso_utc']}] cpu={payload['cpu_usage_pct']:.1f}% ram={payload['ram_pct']:.1f}%")

            elapsed = time.perf_counter() - t0
            if interval_s - elapsed > 0:
                time.sleep(interval_s - elapsed)
    except KeyboardInterrupt:
        print("\nProgram 1 stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
