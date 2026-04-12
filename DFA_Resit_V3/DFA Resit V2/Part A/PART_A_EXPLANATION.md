# Part A - DSMS explanation

This folder contains the four required MQTT programs for the Data Stream Management System part of the resit.

## Folder contents

```text
Part A/
├── PART_A_EXPLANATION.md
└── dsms/
    ├── prog1.py
    ├── prog2.py
    ├── prog3.py
    └── prog4.py
```

## What each program does

### Program 1 - CPU publisher
`dsms/prog1.py`

Publishes Raspberry Pi or host performance data over MQTT at about every 10 ms, depending on what the device can sustain.

Published fields include:
- timestamp in milliseconds
- ISO UTC timestamp
- CPU usage
- CPU frequency
- RAM usage
- context switches
- CPU temperature when available

This is the main stream used by the subscriber programs.

### Program 2 - Sense HAT publisher and location prediction
`dsms/prog2.py`

Reads the Sense HAT accelerometer and publishes:
- raw sensor measurements
- a predicted position and velocity stream

The prediction block uses a small linear Kalman filter with a simple stationary check and a zero-velocity update. That keeps Program 2 aligned with the assignment requirement that the Sense HAT publisher also sends a predicted location.

### Program 3 - Exact rolling-average subscriber
`dsms/prog3.py`

Subscribes to the CPU topic and computes an exact rolling average over a configurable time window.

Example:

```bash
python dsms/prog3.py --window 1.0
python dsms/prog3.py --window 5.0
```

Running two instances with different windows directly addresses the assignment requirement.

### Program 4 - Bernoulli-sampled subscriber
`dsms/prog4.py`

Subscribes to the same CPU topic, but keeps only a random Bernoulli sample of the incoming messages. By default it keeps about one third of the data.

Example:

```bash
python dsms/prog4.py --window 1.0 --p-keep 0.33
```

This gives an approximate rolling average and shows the DSMS trade-off between exact and approximate stream processing.

## Why the architecture is defensible

The architecture is decoupled. Publishers only send messages to MQTT topics and subscribers only listen to topics. The broker handles routing. That makes the design modular and easy to extend.

The reason MQTT fits well here is that it is lightweight, event-driven, widely used in IoT, and built exactly for pub/sub communication between loosely coupled programs.

Using both Program 3 and Program 4 strengthens the design discussion:
- Program 3 gives the exact rolling average for the selected window.
- Program 4 gives an approximate rolling average with lower processing load.

That is a clear and defendable DSMS comparison.

## Two simple malfunction detection rules

These rules can be described as the expert-system part of Part A.

### Rule 1 - Data-gap rule
If the CPU publisher is expected to send a message every 10 ms, but no message arrives for a much longer period, the stream is considered faulty.

Interpretation: the publisher, the broker connection, or the Raspberry Pi may be malfunctioning.

### Rule 2 - Overload rule
If CPU usage stays very high for several seconds, or temperature remains above a chosen safety threshold, the device is considered overloaded.

Interpretation: the Pi may be throttling, unstable, or unable to keep up with the expected stream workload.

## Configuration

The MQTT broker settings, topics, and default window lengths are stored in:

```text
IMU assignment/config/dsms_config.yaml
```

That means the important parameters can be changed without editing the Python code itself.
