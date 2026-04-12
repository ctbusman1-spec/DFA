from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from utils.config_utils import load_yaml


def iso_utc_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_dsms_config(path: str = "config/dsms_config.yaml") -> dict:
    return load_yaml(path)


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
