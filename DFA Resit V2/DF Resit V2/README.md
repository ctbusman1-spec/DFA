# DF Resit V2

This package combines the IMU resit work and Part A (DSMS) in one clean project layout.

The project now has:
- **one `src/` directory** for all code
- **one `data/` directory** for floor plans, experiment logs, and output figures
- **robust path handling** from the project root
- **an improved house floorplan PDF**
- **clearer trajectory overlays**
- **Part A back inside the project** with better configuration and explanations
- **a linear Kalman filter module** kept for assignment-compliant analysis

## Project layout

```text
DF Resit V2/
├── config/
│   ├── system_config.yaml
│   └── dsms_config.yaml
├── data/
│   ├── experiments/
│   └── floor_plans/
├── docs/
│   └── PART_A_EXPLANATION.md
├── src/
│   ├── main.py
│   ├── analysis/
│   │   └── linear_kalman_filter.py
│   ├── dsms/
│   │   ├── prog1.py
│   │   ├── prog2.py
│   │   ├── prog3.py
│   │   └── prog4.py
│   ├── imu/
│   │   ├── filters/
│   │   ├── mapping/
│   │   └── sensors/
│   └── utils/
└── requirements.txt
```

## IMU quick start

Install requirements:

```bash
python -m pip install -r requirements.txt
```

Rebuild the house floorplan PDF and diagnostic images:

```bash
python src/imu/mapping/create_house_floorplan.py
```

Run both IMU models offline:

```bash
python src/main.py
```

Run only one model:

```bash
python src/main.py --filter particle
python src/main.py --filter discrete_bayes
```

Outputs are written to `data/experiments/`.

## Part A quick start

See `docs/PART_A_EXPLANATION.md` for the improved explanation.

Examples:

```bash
python src/dsms/prog1.py
python src/dsms/prog2.py
python src/dsms/prog3.py --window 1.0
python src/dsms/prog3.py --window 5.0
python src/dsms/prog4.py --window 1.0 --p-keep 0.33
```

## Important assignment note

The IMU part now keeps:
- a **Discrete Bayes filter**
- a **Particle filter**
- a **linear Kalman filter module** for analysis / sensor filtering support

So the project is aligned with the assignment requirement that at least one linear Kalman filter remains part of the analysis, while the main localization comparison focuses on the two requested Bayesian approaches.
