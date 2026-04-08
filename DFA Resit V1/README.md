# DFA resubmission codebase

This project rewrites the IMU part around the two models your teacher asked for:
- Particle Filter
- Discrete Bayesian Filter

It removes the Kalman filter from the localization pipeline.

## What this project fixes

- The floor plan is converted into a mathematical probability distribution.
- The motion model includes directional persistence, so forward motion is more likely than random turning.
- Both filters use the same step-based input format.
- The output is not just a line, but also an occupancy heat map.
- The results include a direct model comparison.

## Project layout

```text
config/system_config.yaml
src/main.py
src/filters/particle_filter.py
src/filters/discrete_bayes_filter.py
src/mapping/floor_map.py
src/mapping/create_football_field_map.py
src/sensors/imu_reader.py
src/sensors/stride_detector.py
src/utils/data_logger.py
src/utils/plotting.py
```

## Run on laptop or desktop in offline mode

From the project root:

```bash
python -m pip install -r requirements.txt
python src/mapping/create_house_floorplan.py
python src/main.py
```

That runs both models on the existing CSV file and writes:
- one CSV per model
- one trajectory + heat map image per model
- one summary CSV
- one comparison figure

All outputs go to `src/data/experiments/`.

## Run only one model

```bash
python src/main.py --filter particle
python src/main.py --filter discrete_bayes
```

## Run on Raspberry Pi with Sense HAT

1. Change `mode: offline` to `mode: live` in `config/system_config.yaml`
2. Make sure `sense-hat` works on the Pi
3. Run:

```bash
python src/mapping/create_house_floorplan.py
python src/main.py --filter particle
```

## Important note for the report

Use the code like this in your explanation:
- `FloorMap` converts geometry into a probability distribution `p(x,y | FP)`
- `ParticleFilter` approximates the posterior with weighted samples
- `DiscreteBayesFilter` approximates the posterior on a fixed grid
- both filters use a directional motion prior and map likelihood

## Mathematical framing to match the code

### Particle filter

```text
w_k^(i) ∝ p(x_k^(i), y_k^(i) | FP) · p(x_k^(i) | x_(k-1)^(i), Δθ_k, d_k)
```

### Discrete Bayesian filter

```text
bel_k(c) ∝ p(c | FP) · Σ_c' p(c | c', Δθ_k, d_k) · bel_(k-1)(c')
```

where:
- `FP` is the floor-plan distribution
- `Δθ_k` is heading change
- `d_k` is step length
- `c` is a grid cell

## What to change first

- Replace the football field map with your final field or floor geometry if needed
- Tune `fixed_step_length_m`, `heading_noise_std_rad`, and `directional_persistence`
- Add your final result plots into the report
