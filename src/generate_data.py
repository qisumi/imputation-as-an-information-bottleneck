# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu
# SPDX-License-Identifier: MIT
# Source: https://github.com/nnzhan/Graph-WaveNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

ETT_DATASETS = {"etth1", "etth2", "ettm1", "ettm2"}

def _load_ecg5000_txt(txt_path):
    df = pd.read_csv(txt_path, sep=r"[\s,]+", header=None, engine="python")
    if df.shape[1] <= 1:
        raise ValueError(f"ECG5000 txt file has no usable signal columns: {txt_path}")

    return df.iloc[:, 1:]

def _load_ecg_dataframe(dataset_filename):
    filename = os.path.basename(dataset_filename)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".txt" and "ecg5000" in stem.lower():
        pair_path = None
        ordered_paths = [dataset_filename]
        if stem.endswith("_TRAIN"):
            pair_path = dataset_filename.replace("_TRAIN", "_TEST")
        elif stem.endswith("_TEST"):
            pair_path = dataset_filename.replace("_TEST", "_TRAIN")
            ordered_paths = [pair_path, dataset_filename]

        frames = []
        seen_paths = set()
        for path in ordered_paths:
            if path not in seen_paths and os.path.exists(path):
                frames.append(_load_ecg5000_txt(path))
                seen_paths.add(path)
        if pair_path is not None and pair_path not in seen_paths and os.path.exists(pair_path):
            frames.append(_load_ecg5000_txt(pair_path))
        return pd.concat(frames, ignore_index=True)

    return pd.read_csv(dataset_filename, delimiter=",", header=None)

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):

    if len(df.shape) == 2:
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
    else:
        num_samples, num_nodes, dims = df.shape
        data = df

    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    x, y = [], []

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...].astype(np.float32)
        y_t = data[t + y_offsets, ...].astype(np.float32)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def _load_ett_dataframe(dataset_filename):
    df = pd.read_csv(dataset_filename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError(
            f"No numeric columns were found in ETT csv file: {dataset_filename}"
        )
    return df

def _load_dataframe(args):
    ds_name_lower = args.ds_name.lower()

    if ds_name_lower == "metr-la":
        return pd.read_hdf(args.dataset_filename)

    if ds_name_lower in ETT_DATASETS:
        return _load_ett_dataframe(args.dataset_filename)

    if ds_name_lower == "ecg":
        df = _load_ecg_dataframe(args.dataset_filename)
    else:
        df = pd.read_csv(args.dataset_filename, delimiter=",", header=None)
    if ds_name_lower == "traffic":
        df = df * 1000
    if ds_name_lower == "ecg":
        df = df * 10
    return df

def generate_train_val_test(args):
    df = _load_dataframe(args)

    x_offsets = np.sort(
        np.concatenate((np.arange(1 - args.length, 1, 1),))
    )

    y_offsets = np.sort(np.arange(1, 1 + args.length, 1))

    add_time_in_day = args.ds_name.lower() == "metr-la"
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]

    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )

    x_test, y_test = x[-num_test:], y[-num_test:]

    os.makedirs(args.output_dir, exist_ok=True)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

def main(args):
    print("Generating training data")
    if args.ds_name.lower() not in args.output_dir.lower():
        raise Exception("Incorrect output directory")
    generate_train_val_test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_name", type=str, default="metr-la", help="dataset name."
    )
    parser.add_argument(
        "--output_dir", type=str, default="rawdata/", help="Output directory."
    )
    parser.add_argument(
        "--length", type=int, default=12
    )
    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw dataset readings.",
    )
    args = parser.parse_args()
    main(args)
