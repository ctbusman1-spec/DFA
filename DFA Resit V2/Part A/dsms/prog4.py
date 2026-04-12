#!/usr/bin/env python3
from __future__ import annotations


import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
import random
import time
from collections import deque

from dsms.common import build_client, load_dsms_config


window = deque()


def prune(now: float, window_seconds: float):
    cutoff = now - window_seconds
    while window and window[0][0] < cutoff:
        window.popleft()


def main():
    parser = argparse.ArgumentParser(description="Program 4: Bernoulli-sampled rolling average subscriber.")
    parser.add_argument("--config", default="config/dsms_config.yaml")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--p-keep", type=float, default=None)
    args = parser.parse_args()

    cfg = load_dsms_config(args.config)
    broker = args.broker or cfg["mqtt"]["broker"]
    port = args.port or cfg["mqtt"]["port"]
    topic = args.topic or cfg["topics"]["cpu"]
    window_seconds = args.window or cfg["program4"]["window_seconds"]
    print_every_seconds = cfg["program4"]["print_every_seconds"]
    p_keep = args.p_keep if args.p_keep is not None else cfg["program4"]["p_keep"]

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"Connected. Subscribing to {topic} | window={window_seconds}s | p={p_keep:.2f}")
            client.subscribe(topic, qos=0)
        else:
            print(f"Connect failed: {reason_code}")

    def on_message(client, userdata, msg):
        if random.random() > p_keep:
            return
        now = time.time()
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            cpu_val = float(payload.get("cpu_usage_pct", 0.0))
            window.append((now, cpu_val))
            prune(now, window_seconds)
        except Exception as e:
            print(f"Error processing message: {e}")

    client = build_client(on_connect=on_connect, on_message=on_message)
    client.connect(broker, port, 60)
    client.loop_start()
    last_print = 0.0

    try:
        while True:
            now = time.time()
            prune(now, window_seconds)
            if now - last_print >= print_every_seconds:
                last_print = now
                if not window:
                    print("... waiting for sampled data")
                else:
                    avg_cpu = sum(v for _, v in window) / len(window)
                    print(f"[Approx (Bernoulli) | p={p_keep:.2f} | window={window_seconds:>4.1f}s | n={len(window):>4}] avg CPU = {avg_cpu:6.2f}%")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nProgram 4 stopped.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
