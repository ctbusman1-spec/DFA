# Part A explanation

This file was added to make Part A easier to explain in the report and easier to defend during grading.

## Goal of Part A

Part A is a small data stream management system built around MQTT.

The design is intentionally **decoupled**:
- publishers do not know who subscribes
- subscribers do not need to know how publishers collect data
- the broker sits in the middle and routes all messages

That makes the architecture easy to extend, which is one of the reasons MQTT is a good fit here.

## Program overview

### Program 1 - CPU performance publisher
`src/dsms/prog1.py`

This script reads Raspberry Pi / host performance metrics with `psutil` and publishes them to MQTT every ~10 ms if the machine can keep up.

Published values include:
- timestamp in milliseconds
- ISO-UTC timestamp
- CPU usage percentage
- CPU frequency
- RAM usage percentage
- context switches
- CPU temperature when available

This is the main performance stream used by Programs 3 and 4.

### Program 2 - Sense HAT publisher + Bayesian prediction
`src/dsms/prog2.py`

This script reads the Sense HAT accelerometer and publishes:
- a raw sensor stream
- a predicted location stream

The prediction uses a small linear Kalman filter with a stationary check:
- predict from acceleration
- if the device is stationary, apply a zero-velocity update

This satisfies the assignment requirement that the program publishes Sense HAT readings and a Bayesian prediction.

### Program 3 - Exact rolling average subscriber
`src/dsms/prog3.py`

This subscriber listens to the CPU topic and keeps a time window of messages.

The mean CPU usage is then computed over the full window.  
The window size can be changed from the command line, so it is easy to start two instances with different windows, for example:

```bash
python src/dsms/prog3.py --window 1.0
python src/dsms/prog3.py --window 5.0
```

That directly addresses the assignment requirement to compare different averaging windows.

### Program 4 - Bernoulli-sampled subscriber
`src/dsms/prog4.py`

This subscriber uses the same rolling-window idea as Program 3, but first applies Bernoulli sampling.  
Each incoming message is kept with probability `p_keep`, which is about 1/3 by default.

This gives an approximate average with fewer data points and lower processing effort.

Example:

```bash
python src/dsms/prog4.py --window 1.0 --p-keep 0.33
```

## Why this architecture is defensible

This design scores well because it is not just working code. It also makes architectural sense.

### Why MQTT
MQTT is a strong fit because:
- it is lightweight
- it supports publish/subscribe
- it is common in IoT systems
- producers and consumers remain decoupled

### Why exact and approximate subscribers
Using both Program 3 and Program 4 makes the comparison meaningful:
- Program 3 is the exact streaming average over the selected window
- Program 4 shows what happens when only about one third of the stream is processed

That is a clean DSMS comparison between exact and approximate processing.

## Two malfunction detection rules

Two simple expert-system rules can be described in the report:

### Rule 1 - Data gap detection
If no CPU message arrives for longer than an expected threshold, the Raspberry Pi or publisher is likely malfunctioning.

Example reasoning:
- expected stream interval is 10 ms
- if no message arrives for much longer, something is wrong

### Rule 2 - Thermal or CPU overload
If CPU temperature or CPU usage remains above a high threshold for a sustained period, the device may be overloaded or unstable.

Example reasoning:
- CPU usage > 95% for several seconds
- or CPU temperature above a safety threshold

This is simple model-based reasoning: the subscriber interprets the stream against predefined system rules.

## Why the new version is better than the old Part A integration

This V2 version improves Part A in three ways:
1. The broker, topics, and timing are now in `config/dsms_config.yaml`
2. Programs 3 and 4 support command-line window changes, which matches the assignment more directly
3. The explanation is now explicit, so the architecture and reasoning are easier to present in the report
