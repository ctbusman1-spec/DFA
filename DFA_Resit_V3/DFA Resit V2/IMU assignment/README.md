# IMU assignment

This folder contains the cleaned IMU resit submission: code, configuration, logs, output figures, and a final notebook that reads the already generated results.

## Folder structure

```text
IMU assignment/
├── Analysis/
│   ├── final_results_summary.py
│   └── run_linear_kalman.py
├── Final_Report.ipynb
├── README.md
├── config/
│   ├── dsms_config.yaml
│   └── system_config.yaml
├── data/
│   ├── experiments/
│   ├── floor_plans/
│   └── logs/
├── requirements.txt
└── src/
    ├── main.py
    ├── analysis/
    │   └── linear_kalman_filter.py
    ├── imu/
    │   ├── filters/
    │   ├── mapping/
    │   └── sensors/
    └── utils/
```

## What is included

This submission contains the three methods required for the report:
- **Discrete Bayes filter** for map-based localization
- **Particle filter** for recursive Bayesian localization
- **Linear Kalman filter** for sensor-level smoothing analysis

The localization comparison is between the Discrete Bayes filter and the Particle filter. The Kalman filter is kept as the required linear filter in the analysis section.

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Rebuild the floorplan PDF if needed:

```bash
python src/imu/mapping/create_house_floorplan.py
```

Run the offline IMU experiment configured in `config/system_config.yaml`:

```bash
python src/main.py
```

Run only one localization model:

```bash
python src/main.py --filter particle
python src/main.py --filter discrete_bayes
```

Run the linear Kalman analysis on the raw sensor logs:

```bash
python Analysis/run_linear_kalman.py
```

Summarize all saved experiment results:

```bash
python Analysis/final_results_summary.py
```

## Final report notebook

The notebook below is the report-style notebook for submission:

```text
Final_Report.ipynb
```

It loads the existing CSV summaries, configuration values, code references, and generated figures from the `data/experiments/` folder.

## Important files

### Main localization logic
- `src/imu/filters/discrete_bayes_filter.py`
- `src/imu/filters/particle_filter.py`
- `src/main.py`

### Linear Kalman filter
- `src/analysis/linear_kalman_filter.py`
- `Analysis/run_linear_kalman.py`

### Spatial alignment and floorplan PDF
- `src/imu/mapping/floor_map.py`
- `src/imu/mapping/create_house_floorplan.py`
- `data/floor_plans/house_floorplan_pdf.pkl`

### Output data already included
- run summaries and plots in `data/experiments/`
- aggregated summaries in `data/experiments/outputs/`

## Notes for interpretation

There is no external motion-capture ground truth in this submission. The evaluation therefore focuses on:
- floorplan consistency
- walkable ratio
- trajectory plausibility
- runtime per step
- repeatability across several walk, turn, and still logs

