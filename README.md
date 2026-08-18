# Imputation as an Information Bottleneck

Official implementation of **“Imputation as an Information Bottleneck: Direct Mask-Conditioned Forecasting for Missing Time Series,” accepted by IEEE ICDM 2026**.

This repository contains source code, experiment scripts, dependency metadata, and data-preparation instructions.

## Repository layout

- `src/`: data conversion, missingness construction, imputation baselines, MEM, forecasting backbones, and evaluation.
- `scripts/preparation/`: clean-dataset conversion and missing/imputed test-set preparation.
- `scripts/benchmarks/`: main experiments and paper ablations.
- `scripts/workflows/`: end-to-end experiment drivers.
- `requirements.txt`: Python dependencies other than the platform-specific CUDA runtime.

Runtime outputs are written under `rawdata/`, `save*/`, and `runlog/`. All three locations are ignored by Git.

## Environment

Python 3.11 and PyTorch 2.x are recommended. Create an isolated environment, install the PyTorch build appropriate for your CPU or CUDA runtime, and then install the remaining dependencies:

```sh
python -m venv .venv
source .venv/bin/activate

pip install torch
pip install -r requirements.txt
```

GPU scripts default to `cuda:0` or the device recorded in the archived experiment script. Review `DEVICE` and the command-line options before starting a full run.

## Public datasets

The repository does not redistribute third-party data. Download the four dataset families from their public sources and use the exact filenames below.

| Dataset | Public source | Expected raw files |
| --- | --- | --- |
| ECG5000 | [UCR/UEA dataset page](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) and [archive download](https://www.timeseriesclassification.com/Downloads/ECG5000.zip) | `rawdata/ECG5000/ECG5000_TRAIN.txt`, `rawdata/ECG5000/ECG5000_TEST.txt` |
| METR-LA | [DCRNN data instructions](https://github.com/liyaguang/DCRNN#data-preparation) and its [public Google Drive folder](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) | `rawdata/metr-la.h5` |
| Solar-Energy | [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data/tree/master/solar-energy) | `rawdata/solar_AL.txt` |
| ETT | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | `rawdata/ETTh1/ETTh1.csv`, `rawdata/ETTh2/ETTh2.csv`, `rawdata/ETTm1/ETTm1.csv`, `rawdata/ETTm2/ETTm2.csv` |

### ECG5000

Download and extract ECG5000, then keep the UCR text files in a source directory:

```sh
mkdir -p rawdata/ECG5000
unzip /path/to/ECG5000.zip -d rawdata/ECG5000
```

`src/generate_data.py` reads both `ECG5000_TRAIN.txt` and `ECG5000_TEST.txt`, removes the first classification-label column, concatenates the signals, applies the archived ECG scaling, and constructs chronological forecasting windows.

### METR-LA

Download `metr-la.h5` from the DCRNN data folder and place it at `rawdata/metr-la.h5`. The file must be a pandas HDF5 DataFrame with a datetime index and one column per traffic sensor.

### Solar-Energy

The source repository provides a gzip-compressed comma-separated matrix. Download and decompress it as follows:

```sh
mkdir -p rawdata
curl -L \
  https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz \
  -o rawdata/solar_AL.txt.gz
gzip -dc rawdata/solar_AL.txt.gz > rawdata/solar_AL.txt
```

### ETT

The four CSV files can be downloaded directly from the official ETT repository:

```sh
for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
  mkdir -p "rawdata/${dataset}"
  curl -L \
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/${dataset}.csv" \
    -o "rawdata/${dataset}/${dataset}.csv"
done
```

## Convert raw data

After all raw files are in place, generate the clean chronological splits:

```sh
bash scripts/preparation/datasets.sh
```

The converter creates input/output windows with a 70%/10%/20% chronological train/validation/test split. The main datasets use input and prediction horizons of 12. ECG horizon experiments additionally create `ECG24`, `ECG36`, and `ECG48` directories. Generated `.npz` files remain under ignored `rawdata/` directories.

Equivalent individual commands are:

```sh
python src/generate_data.py \
  --ds_name ecg --length 12 \
  --dataset_filename rawdata/ECG5000/ECG5000_TRAIN.txt \
  --output_dir rawdata/ECG

python src/generate_data.py \
  --ds_name metr-la --length 12 \
  --dataset_filename rawdata/metr-la.h5 \
  --output_dir rawdata/METR-LA

python src/generate_data.py \
  --ds_name solar --length 12 \
  --dataset_filename rawdata/solar_AL.txt \
  --output_dir rawdata/SOLAR
```

## Construct missing data and imputation baselines

Prepare the MCAR 10%, 20%, and 30% test sets and all five imputation baselines:

```sh
DEVICE=cuda:0 IMPUTATION_EPOCHS=100 \
  bash scripts/preparation/main_mcar.sh
```

Prepare the MAR- and MNAR-labelled ECG and METR-LA variants used by the missing-pattern experiment:

```sh
DEVICE=cuda:0 IMPUTATION_EPOCHS=100 \
  bash scripts/preparation/missing_patterns.sh
```

These commands create missing and imputed `.npz` files beside each clean dataset.

Missingness construction and every imputer use the fixed preparation seed `42`, producing one shared missing-data realization for strict method comparison. Forecasting independently uses the five default training seeds `42 2023 2024 2025 2026`.

## Run the experiments

### Main MCAR comparison

Each benchmark trains the clean-data Oracle first. Every imputation baseline then loads the matching Oracle checkpoint for the same random seed and evaluates it on that method's imputed test input.

For Solar-Energy, zero-valued targets represent the timestamp-linked nighttime regime. Following the archived dataset-specific protocol, zero targets are excluded from MAE, RMSE, and MAPE aggregation rather than being treated as ordinary daytime forecasting targets.

```sh
bash scripts/benchmarks/mcar_10.sh
bash scripts/benchmarks/mcar_20.sh
bash scripts/benchmarks/mcar_30.sh
```

The complete preparation-plus-benchmark workflow is:

```sh
bash scripts/workflows/main_mcar.sh
```

### Missing patterns

```sh
bash scripts/workflows/missing_patterns.sh
```

### MEM encoder and forecasting-backbone ablations

```sh
bash scripts/benchmarks/mem_encoders.sh
bash scripts/benchmarks/backbone_oracle.sh
bash scripts/benchmarks/backbone_mem.sh
```

### ECG horizons and ETT transfer

The ECG workflow reuses the ECG rows from the MCAR-20% main result and evaluates horizons 24, 36, and 48:

```sh
STEP12_ARCHIVE=runlog/pred_with_imputation_bench.csv \
  bash scripts/workflows/ecg_horizons.sh

bash scripts/workflows/ett.sh
```

See `scripts/README.md` for the mapping between scripts and experiment groups.

## Small smoke test

After preparing at least the ECG dataset, the following commands exercise clean-data forecasting and MEM without running the full matrix:

```sh
python src/data_imputation.py \
  --datasets ECG --methods BRITS --epochs 1 \
  --drop_rate 0.2 --missing_type MCAR --device cuda:0 \
  --result_path runlog/smoke_imputation.csv

python src/run_mtgnn.py \
  --dataset ECG --expid 10 --seeds 42 --epochs 1 --device cuda:0 \
  --result_path runlog/smoke_oracle.csv

python src/run_mtgnn.py \
  --dataset ECG --imputation_method BRITS --seeds 42 --device cuda:0 \
  --result_path runlog/smoke_brits.csv

python src/run_mtgnn.py \
  --dataset ECG --mask_layer True --seeds 42 --epochs 1 --device cuda:0 \
  --result_path runlog/smoke_mem.csv
```

## License and attribution

The project is released under the [MIT License](LICENSE). Upstream copyright, license, and source information for Graph WaveNet-derived data preparation and MTGNN-derived model code is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and retained in the relevant source headers.
