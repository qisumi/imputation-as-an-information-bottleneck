import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from pypots.imputation import SAITS, MRNN, BRITS, Transformer, USGAN
from pypots.utils.metrics import calc_mae, calc_rmse

from utils.utils import generate_mask_mat

DEFAULT_DATASETS = ["ECG", "METR-LA", "SOLAR"]
DEFAULT_METHODS = ["SAITS", "MRNN", "BRITS", "Transformer", "USGAN"]
FIXED_PREPARATION_SEED = 42

def seed_preparation():
    random.seed(FIXED_PREPARATION_SEED)
    np.random.seed(FIXED_PREPARATION_SEED)
    torch.manual_seed(FIXED_PREPARATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FIXED_PREPARATION_SEED)
        torch.cuda.manual_seed_all(FIXED_PREPARATION_SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _build_model_instances(X, device, epochs):
    return {
        "SAITS": SAITS(
            n_steps=X.shape[1],
            n_features=X.shape[2],
            n_layers=2,
            d_model=256,
            d_ffn=128,
            n_heads=4,
            d_k=64,
            d_v=64,
            dropout=0.1,
            epochs=epochs,
            num_workers=4,
            device=device,
        ),
        "MRNN": MRNN(
            n_steps=X.shape[1],
            n_features=X.shape[2],
            rnn_hidden_size=256,
            epochs=epochs,
            num_workers=4,
            device=device,
        ),
        "BRITS": BRITS(
            n_steps=X.shape[1],
            n_features=X.shape[2],
            rnn_hidden_size=256,
            epochs=epochs,
            num_workers=4,
            device=device,
        ),
        "Transformer": Transformer(
            n_steps=X.shape[1],
            n_features=X.shape[2],
            n_layers=2,
            d_model=256,
            d_ffn=128,
            n_heads=4,
            d_k=64,
            d_v=64,
            dropout=0.1,
            epochs=epochs,
            num_workers=4,
            device=device,
        ),
        "USGAN": USGAN(
            n_steps=X.shape[1],
            n_features=X.shape[2],
            rnn_hidden_size=256,
            dropout=0.1,
            epochs=epochs,
            num_workers=4,
            device=device,
        ),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--missing_type', type=str, default='MCAR')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=DEFAULT_DATASETS,
        help='datasets to process, e.g. ECG METR-LA SOLAR ETTh1 ETTh2 ETTm1 ETTm2 ECG24',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=DEFAULT_METHODS,
        help='imputation methods to run',
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_root', type=str, default='rawdata')
    parser.add_argument('--result_path', type=str, default=None)

    args = parser.parse_args()
    drop_rate = args.drop_rate
    missing_type = args.missing_type.upper()
    suffix = 'mcar' if missing_type == 'MCAR' else missing_type.lower()
    device = torch.device(args.device)

    if missing_type == 'MCAR':
        testset_path = '{data_root}/{dataset}/test_mcar_{drop_rate}.npz'
    else:
        testset_path = f'{{data_root}}/{{dataset}}/test_{suffix}_{{drop_rate}}.npz'

    if missing_type == 'MCAR':
        save_path = '{data_root}/{dataset}/test_{method}_{drop_rate}.npz'
        csv_path = f'runlog/imputation_nn_{drop_rate}.csv'
    else:
        save_path = f'{{data_root}}/{{dataset}}/test_{{method}}_{suffix}_{{drop_rate}}.npz'
        csv_path = f'runlog/imputation_nn_{suffix}_{drop_rate}.csv'

    if args.result_path is not None:
        csv_path = args.result_path

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    result_table = pd.DataFrame(
        columns=['ImputationMethod', 'Dataset', 'MAE', 'RMSE']
    )

    for dataset in args.datasets:
        test_file = os.path.join(args.data_root, dataset, 'test.npz')
        if not os.path.exists(test_file):
            raise FileNotFoundError(
                f'Missing test set for dataset {dataset}: {test_file}'
            )

        rawdata = np.load(test_file)
        X_full = rawdata['x']

        seed_preparation()
        mask = generate_mask_mat(X_full[..., 0], p=drop_rate, missing_type=missing_type)
        X_missing = X_full.copy()
        X_missing[..., 0][mask] = np.nan

        X = X_missing[:, :, :, 0]

        np.savez(
            testset_path.format(data_root=args.data_root, dataset=dataset, drop_rate=drop_rate),
            x=X_missing,
            y=rawdata['y'],
            x_offsets=rawdata['x_offsets'],
            y_offsets=rawdata['y_offsets'],
        )

        X_intact = X_full[:, :, :, 0]
        indicating_mask = mask.astype(float)
        for model in args.methods:
            seed_preparation()
            model_instances = _build_model_instances(X, device, args.epochs)
            if model not in model_instances:
                raise ValueError(f'Unsupported imputation method: {model}')

            imputation_model = model_instances[model]
            imputation_model.fit({"X": X})
            imputation = imputation_model.impute({"X": X})

            mae = calc_mae(imputation, X_intact, indicating_mask)
            rmse = calc_rmse(imputation, X_intact, indicating_mask)
            result_table = pd.concat([result_table, pd.DataFrame([{
                'ImputationMethod': model,
                'Dataset': dataset,
                'MAE': f"{mae}",
                'RMSE': f"{rmse}"
            }])], ignore_index=True)
            result_table.to_csv(csv_path, index=False)

            save_data = np.load(testset_path.format(
                data_root=args.data_root,
                dataset=dataset,
                drop_rate=drop_rate,
            ))
            if save_data['x'].shape[3] == 2:
                imputation = np.concatenate(
                    [
                        imputation[:, :, :, np.newaxis],
                        save_data['x'][..., 1][:, :, :, np.newaxis],
                    ],
                    axis=3,
                )
            else:
                imputation = imputation[:, :, :, np.newaxis]
            np.savez(
                save_path.format(
                    data_root=args.data_root,
                    dataset=dataset,
                    method=model,
                    drop_rate=drop_rate,
                ),
                x=imputation,
                y=save_data['y'],
                x_offsets=save_data['x_offsets'],
                y_offsets=save_data['y_offsets'],
            )
