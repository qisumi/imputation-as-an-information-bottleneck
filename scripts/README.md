# Experiment scripts

All scripts resolve the repository root before invoking Python. Generated datasets, checkpoints, logs, and CSV files are runtime outputs and are ignored by Git.

## Preparation

- `preparation/datasets.sh`: convert public raw files into clean chronological `.npz` splits, including ECG horizons.
- `preparation/main_mcar.sh`: construct MCAR 10%, 20%, and 30% inputs and imputation baselines.
- `preparation/missing_patterns.sh`: construct MAR- and MNAR-labelled ECG and METR-LA inputs.
- `preparation/ecg_horizons.sh`: impute the prepared ECG24/36/48 test sets.
- `preparation/ett.sh`: convert the four ETT CSV files.
- `preparation/ett_imputation.sh`: construct ETT MCAR-20% inputs and imputation baselines.

## Benchmarks

- `benchmarks/mcar_10.sh`, `mcar_20.sh`, `mcar_30.sh`: main missing-rate comparison.
- `benchmarks/missing_patterns.sh`: MCAR/MAR/MNAR-labelled comparison on ECG and METR-LA.
- `benchmarks/mem_encoders.sh`: Conv2D, MLP, and Attention MEM variants.
- `benchmarks/backbone_oracle.sh`, `backbone_mem.sh`: DLinear and TQNet compatibility experiment.
- `benchmarks/ecg_horizons.sh`: ECG prediction-horizon experiment.
- `benchmarks/ett_mcar20.sh`: ETT transfer experiment.

## Workflows

- `workflows/main_mcar.sh`: prepare MCAR inputs and run the three main missing-rate matrices.
- `workflows/missing_patterns.sh`: prepare and run the missing-pattern matrix.
- `workflows/ecg_horizons.sh`: prepare, run, deduplicate, and summarize the ECG horizon matrix.
- `workflows/ett.sh`: convert ETT data, run imputation, and run forecasting.

All forecasting scripts explicitly pass `SEEDS`, defaulting to `42 2023 2024 2025 2026`. Supply a whitespace-separated override such as `SEEDS="7 8 9" bash scripts/benchmarks/mcar_20.sh`. Review device, epoch, and result-path settings before launching a full matrix.
