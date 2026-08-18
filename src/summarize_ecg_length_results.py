import argparse
import re

import pandas as pd

MODEL_ORDER = ["Oracle", "BRITS", "MRNN", "SAITS", "Transformer", "USGAN", "proposed"]

def _normalize_dataset_name(dataset_name: str):
    dataset_name = str(dataset_name)
    match = re.match(r'^(.*?)(\d+)?$', dataset_name)
    if match is None:
        return dataset_name, None
    base_dataset = match.group(1)
    seq_len = match.group(2)
    seq_len = int(seq_len) if seq_len is not None else None
    return base_dataset, seq_len

def _load_result_table(csv_path: str, default_seq_len=None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["dataset", "imputation_model"])

    base_datasets = []
    seq_lens = []
    for dataset_name in df["dataset"]:
        base_dataset, seq_len = _normalize_dataset_name(dataset_name)
        base_datasets.append(base_dataset)
        seq_lens.append(default_seq_len if seq_len is None else seq_len)

    df = df.copy()
    df["base_dataset"] = base_datasets
    df["seq_len"] = seq_lens
    df = df[df["base_dataset"] == "ECG"]
    if default_seq_len is not None:
        df["seq_len"] = default_seq_len
    df = df[df["seq_len"].isin([12, 24, 36, 48])]
    return df

def _sort_results(df: pd.DataFrame):
    model_rank = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    df = df.copy()
    df["_model_rank"] = df["imputation_model"].map(model_rank).fillna(len(MODEL_ORDER))
    df = df.sort_values(["seq_len", "_model_rank", "pred_model", "imputation_model"])
    return df.drop(columns=["_model_rank"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step12_result",
        type=str,
        default="runlog/pred_with_imputation_bench.csv",
        help="step-12 benchmark CSV",
    )
    parser.add_argument(
        "--length_result",
        type=str,
        default="runlog/ecg_horizons_mcar20.csv",
        help="new ECG length benchmark csv",
    )
    parser.add_argument(
        "--output_long",
        type=str,
        default="runlog/ecg_horizons_mcar20_long.csv",
    )
    parser.add_argument(
        "--output_wide",
        type=str,
        default="runlog/ecg_horizons_mcar20_wide.csv",
    )
    parser.add_argument(
        "--output_step12_csv",
        type=str,
        default=None,
        help="optional path to export normalized step-12 results",
    )
    args = parser.parse_args()

    step12_df = _load_result_table(args.step12_result, default_seq_len=12)
    length_df = _load_result_table(args.length_result)

    if args.output_step12_csv:
        export_columns = ["pred_model", "dataset", "imputation_model", "mae", "rmse", "mape"]
        step12_export = step12_df.copy()
        step12_export["dataset"] = step12_export["base_dataset"]
        step12_export[export_columns].to_csv(args.output_step12_csv, index=False)

    merged = pd.concat([step12_df, length_df], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["pred_model", "base_dataset", "seq_len", "imputation_model"],
        keep="last",
    )
    merged = _sort_results(merged)

    long_columns = [
        "pred_model",
        "base_dataset",
        "seq_len",
        "dataset",
        "imputation_model",
        "mae",
        "rmse",
        "mape",
        "kernel_size",
        "dropout",
    ]
    for column in ["kernel_size", "dropout"]:
        if column not in merged.columns:
            merged[column] = None
    merged[long_columns].to_csv(args.output_long, index=False)

    wide = merged.pivot_table(
        index=["pred_model", "imputation_model"],
        columns="seq_len",
        values=["mae", "rmse", "mape"],
        aggfunc="first",
    )
    wide = wide.sort_index(axis=1, level=[0, 1])
    wide.to_csv(args.output_wide)

    print(f"Saved long-format summary to {args.output_long}")
    print(f"Saved wide-format summary to {args.output_wide}")

if __name__ == "__main__":
    main()
