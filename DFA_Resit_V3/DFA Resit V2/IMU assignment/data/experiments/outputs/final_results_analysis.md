# Final Results Summary

## Overall interpretation

### 180Turn
- Higher mean walkable ratio: **discrete_bayes** (0.969 vs 0.969).
- Runtime per step: discrete_bayes **3.09 ms**, particle **9.10 ms**.

### outputs
- Higher mean walkable ratio: **discrete_bayes** (0.742 vs 0.711).
- Runtime per step: discrete_bayes **2.37 ms**, particle **6.82 ms**.

### shortTurn
- Higher mean walkable ratio: **discrete_bayes** (1.000 vs 0.875).
- Runtime per step: discrete_bayes **2.92 ms**, particle **8.98 ms**.

### still
- Higher mean walkable ratio: **discrete_bayes** (0.000 vs 0.000).
- Runtime per step: discrete_bayes **0.00 ms**, particle **0.00 ms**.

### walk
- Higher mean walkable ratio: **discrete_bayes** (1.000 vs 1.000).
- Runtime per step: discrete_bayes **3.49 ms**, particle **9.22 ms**.

## Best model per run

```text
    group              run_name          model  walkable_ratio  path_length_m  runtime_ms_per_step
  180Turn         log_turn180_1 discrete_bayes          0.9375       9.129097             2.986375
  180Turn         log_turn180_2 discrete_bayes          1.0000       5.753036             3.195364
  outputs         final_results discrete_bayes          1.0000       1.866339             2.677250
  outputs final_results_grouped discrete_bayes          1.0000       2.422563             2.917000
shortTurn       log_short_turn1 discrete_bayes          1.0000       2.978787             3.156750
shortTurn       log_short_turn2 discrete_bayes          1.0000       1.866339             2.677250
    still             log_still       particle          0.0000       0.000000             0.000000
    still            log_still2       particle          0.0000       0.000000             0.000000
     walk             log_walk1 discrete_bayes          1.0000       4.108927             3.900563
     walk             log_walk2 discrete_bayes          1.0000       4.387162             3.080689
```

## Notes

- Still runs should be used mainly for noise / false-step / bias checks, not as trajectory-performance cases.
- Walk runs are the main baseline for step-length and stability.
- Short-turn / walk-turn runs are the most informative for heading behavior and corner handling.
- 180-turn runs are strongest as limitation or challenge cases, not as primary performance figures.