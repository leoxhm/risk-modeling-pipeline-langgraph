# Training and tuning node

Reserved for the standalone LightGBM baseline and bounded Optuna workflow.

When `tuning.cv_strategy: rolling`, the executor builds expanding windows from
the confirmed Train sample only. Every Optuna trial uses the same windows and
its own early-stopping iteration per window. The run writes
`tuning/tuning_trials.csv` (one row per parameter set),
`tuning/cv_fold_metrics.csv` (one row per trial/window, including month ranges),
and `tuning/best_params.yaml`. Test and OOT are evaluated only after tuning and
never affect the Optuna objective.
