# Linear Kalman Filter Analysis

This analysis applies the linear Kalman filter to the first raw sensor log in each experiment category. The Kalman filter is used here as a sensor-level smoothing method on gyroscope signals, not as the main map-based localization model.

## Summary

```text
    group               run_name  duration_s  raw_gyro_z_std  kf_gyro_z_std  raw_final_yaw_change  kf_final_yaw_change
    still      sensor_log_still1   10.043318        0.002877       0.001132             -0.284822            -0.286582
     walk       sensor_log_walk1   13.946918        0.276807       0.087465              0.014902            -0.005268
shortTurn sensor_log_short_turn1   14.081547        0.315779       0.208597              1.530645             1.549969
  180Turn   sensor_log_turn180_1   19.248076        0.565229       0.394996              4.251548             4.182481
```

## Interpretation

- **still** should show low gyroscope variance and low integrated yaw drift.
- **walk** should remain relatively stable with smoother filtered gyro signals.
- **shortTurn** should preserve the main turn while reducing local noise.
- **180Turn** can be used as a challenge case: the Kalman filter smooths the signal, but does not solve the full localization problem on its own.