from models.MTGNN.MTGNN import gtnet as MTGNN
import numpy as np
import torch
from torch import nn
from models.MTGNN.utils import *
from models.MTGNN.MTGNN import gtnet
from models.MTGNN.Trainer import *
from models.MTGNN.DLinear import DLinearBackbone
from models.MTGNN.TQNet import TQNetBackbone
import random
import pandas as pd
import time
import argparse
import os

from utils.dataset_utils import infer_dataset_shape, resolve_pretrained_expid

DEFAULT_SEEDS = [42, 2023, 2024, 2025, 2026]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--dataset', type=str, default='ECG')
    parser.add_argument('--imputation_method', type=str,
                        default=None)
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument(
        '--pretrained_expid',
        type=int,
        default=None,
        help='checkpoint expid to load in skip-training mode; '
             'required for new datasets such as ECG24 / ETTh1 when no legacy default exists',
    )

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help='explicit random seeds; defaults to 42 2023 2024 2025 2026',
    )
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--save', type=str,
                        default='./save/', help='save path')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--seq_in_len', type=int,
                        default=12, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int,
                        default=12, help='output sequence length')
    parser.add_argument('--mask_layer', type=str, default="false")
    parser.add_argument('--mask_backbone', type=str, default='conv2d',
                        help='mask layer backbone: conv2d | mlp | attention')
    parser.add_argument('--missing_type', type=str, default='MCAR')
    parser.add_argument('--test_only', type=str, default="false")
    parser.add_argument('--drop_rate', type=str, default='0.2')
    parser.add_argument('--mask_kernel_size', type=int, default=2)
    parser.add_argument('--mask_dropout', type=float, default=0.25)
    parser.add_argument('--backbone', type=str, default='MTGNN',
                        help='prediction backbone: MTGNN | DLinear | TQNet')
    parser.add_argument('--result_path', type=str, default=None,
                        help='result csv path; default: runlog/pred_with_imputation_bench{drop_rate}.csv')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f'Random seeds must be unique, got {args.seeds}')
    torch.set_num_threads(3)
    args.mask_layer = str(args.mask_layer).strip().lower() in {"true", "1", "yes", "y"}
    args.test_only = str(args.test_only).strip().lower() in {"true", "1", "yes", "y"}
    args.missing_type = args.missing_type.upper()
    args.backbone = args.backbone.upper()
    drop_rate = "" if args.drop_rate == '0.2' else f"_{args.drop_rate}"
    args.save = f"./save{drop_rate}/"
    os.makedirs(args.save, exist_ok=True)
    if args.result_path is None:
        result_path = f'runlog/pred_with_imputation_bench{drop_rate}.csv'
    else:
        result_path = args.result_path
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    datadir = f"rawdata/{args.dataset}"
    num_split = 1
    step_size2 = 100
    num_nodes, in_dim = infer_dataset_shape(datadir)

    def build_model(device, num_nodes, in_dim):
        if args.backbone == 'MTGNN':

            subgraph_size = min(20, num_nodes)
            print(f'Using MTGNN subgraph_size={subgraph_size} for num_nodes={num_nodes}')
            return gtnet(
                gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=num_nodes,
                device=device, predefined_A=None, dropout=0.3, subgraph_size=subgraph_size, node_dim=40,
                dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64,
                end_channels=128, seq_length=args.seq_in_len, in_dim=in_dim, out_dim=args.seq_out_len,
                mask_layer_enable=args.mask_layer, mask_kernel_size=args.mask_kernel_size,
                mask_dropout=args.mask_dropout, mask_backbone=args.mask_backbone
            )
        elif args.backbone == 'DLINEAR':
            return DLinearBackbone(
                seq_len=args.seq_in_len,
                pred_len=args.seq_out_len,
                num_nodes=num_nodes,
                mask_layer_enable=args.mask_layer,
                mask_kernel_size=args.mask_kernel_size,
                mask_dropout=args.mask_dropout,
                mask_backbone=args.mask_backbone,
            )
        elif args.backbone == 'TQNET':
            return TQNetBackbone(
                seq_len=args.seq_in_len,
                pred_len=args.seq_out_len,
                num_nodes=num_nodes,
                mask_layer_enable=args.mask_layer,
                mask_kernel_size=args.mask_kernel_size,
                mask_dropout=args.mask_dropout,
                mask_backbone=args.mask_backbone,
            )
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone}")

    def maybe_load_pretrained_model(model, device, run_seed):
        load_expid = resolve_pretrained_expid(
            dataset=args.dataset,
            mask_layer=args.mask_layer,
            pretrained_expid=args.pretrained_expid,
        )
        checkpoint_path = os.path.join(
            args.save, f'exp{load_expid}_seed{run_seed}.pth'
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Expected pretrained checkpoint {checkpoint_path} does not exist. "
                f"Please run expid={load_expid} first or provide a valid --pretrained_expid."
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'loaded checkpoint: {checkpoint_path}')
        return model

    def main(run_seed):
        global num_nodes
        device = torch.device(args.device)
        if args.mask_layer:
            args.imputation_method = 'mcar'
        dataloader = load_dataset(
            datadir,
            args.batch_size,
            args.batch_size,
            args.batch_size,
            args.imputation_method,
            mask_layer=args.mask_layer,
            drop_rate=drop_rate,
            missing_type=args.missing_type,
            missing_rate=float(args.drop_rate),
        )
        scaler = dataloader['scaler']
        model = build_model(device, num_nodes, in_dim)

        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams)

        should_skip_training = (
            (args.imputation_method is not None and str(args.imputation_method).lower() != 'mcar')
            or args.test_only
        )

        if should_skip_training:
            model = maybe_load_pretrained_model(model, device, run_seed)
            print('skip training...')
            engine = Trainer(model, lrate=0.001, wdecay=0.0001, clip=5, step_size=2500,
                             seq_out_len=args.seq_out_len, scaler=scaler, device=device, cl=True)
        else:
            engine = Trainer(model, lrate=0.001, wdecay=0.0001, clip=5, step_size=2500,
                             seq_out_len=args.seq_out_len, scaler=scaler, device=device, cl=True)
            print('start training...')

            his_loss = []
            val_time = []
            train_time = []
            minl = 1e5
            for i in range(1, args.epochs + 1):
                train_loss = []
                train_mape = []
                train_rmse = []
                t1 = time.time()
                dataloader['train_loader'].shuffle()
                for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                    trainx = torch.Tensor(x).to(device)
                    trainx = trainx.transpose(1, 3)
                    trainy = torch.Tensor(y).to(device)
                    trainy = trainy.transpose(1, 3)
                    if iter % step_size2 == 0:
                        perm = np.random.permutation(range(num_nodes))
                    num_sub = int(num_nodes / num_split)
                    for j in range(num_split):
                        if j != num_split - 1:
                            id = perm[j * num_sub:(j + 1) * num_sub]
                        else:
                            id = perm[j * num_sub:]
                        id = torch.tensor(id).to(device)
                        tx = trainx[:, :, id, :]
                        ty = trainy[:, :, id, :]
                        metrics = engine.train(tx, ty[:, 0, :, :], id)
                        train_loss.append(metrics[0])
                        train_mape.append(metrics[1])
                        train_rmse.append(metrics[2])
                    if iter % args.print_every == 0:
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(
                            iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                t2 = time.time()
                train_time.append(t2 - t1)

                valid_loss = []
                valid_mape = []
                valid_rmse = []

                s1 = time.time()
                for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    metrics = engine.eval(testx, testy[:, 0, :, :])
                    valid_loss.append(metrics[0])
                    valid_mape.append(metrics[1])
                    valid_rmse.append(metrics[2])
                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i, (s2 - s1)))
                val_time.append(s2 - s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                                 mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

                if mvalid_loss < minl:
                    checkpoint_path = os.path.join(
                        args.save,
                        f'exp{args.expid}_seed{run_seed}.pth',
                    )
                    torch.save(engine.model.state_dict(), checkpoint_path)
                    minl = mvalid_loss

            print(
                "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
            bestid = np.argmin(his_loss)
            checkpoint_path = os.path.join(
                args.save, f'exp{args.expid}_seed{run_seed}.pth'
            )
            engine.model.load_state_dict(torch.load(
                checkpoint_path, map_location=device))

            print("Training finished")
            print("The valid loss on best model is",
                  str(round(his_loss[bestid], 4)))

        outputs = []
        realy = torch.Tensor(dataloader['y_val']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        pred = scaler.inverse_transform(yhat)
        vmae, vmape, vrmse = metric(pred, realy)

        outputs = []
        if args.imputation_method is None:

            realy = torch.Tensor(dataloader['y_test']).to(device)
        else:

            mt = args.missing_type.upper()
            if drop_rate:
                file_drop_rate = drop_rate.lstrip('_')
            else:
                file_drop_rate = '0.2'

            if args.mask_layer:

                if mt == "MCAR":
                    test_cat = f'test_mcar_{file_drop_rate}'
                else:
                    test_cat = f'test_{mt.lower()}_{file_drop_rate}'
            else:

                if mt == "MCAR":
                    test_cat = f'test_{args.imputation_method}_{file_drop_rate}'
                else:
                    test_cat = f'test_{args.imputation_method}_{mt.lower()}_{file_drop_rate}'

            realy = torch.Tensor(dataloader[f'y_{test_cat}']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        mae = []
        mape = []
        rmse = []
        for i in range(args.seq_out_len):
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        return vmae, vmape, vrmse, mae, mape, rmse

if __name__ == "__main__":
    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for seed in args.seeds:
        print(f'\n=== Random seed: {seed} ===')
        seed_everything(seed)
        vm1, vm2, vm3, m1, m2, m3 = main(seed)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print(f'\n\nResults for seeds {args.seeds}\n\n')

    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n\n')

    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    test_indices = [2, 5, 11]
    valid_test_indices = [i for i in test_indices if i < args.seq_out_len]
    if len(valid_test_indices) == 0:
        valid_test_indices = list(range(args.seq_out_len))
    for i in valid_test_indices:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, amae[i], armse[i],
              amape[i], smae[i], srmse[i], smape[i]))

    if args.imputation_method is None:
        args.imputation_method = 'Oracle'
    if args.mask_layer:
        args.imputation_method = 'proposed'

    try:
        result_table = pd.read_csv(result_path)
    except FileNotFoundError:
        result_table = pd.DataFrame(columns=[
            'pred_model',
            'dataset',
            'imputation_model',
            'mae',
            'rmse',
            'mape',
            'kernel_size',
            'dropout',
            'seeds',
        ])
    result_table = pd.concat([result_table, pd.DataFrame([{
        'pred_model': args.backbone,
        'dataset': args.dataset,
        "imputation_model": args.imputation_method,
        'mae': f"{round(np.mean(amae), 4)}({round(np.mean(smae), 4)})",
        'rmse': f"{round(np.mean(armse), 4)}({round(np.mean(srmse), 4)})",
        'mape': f"{round(np.mean(amape), 4)}({round(np.mean(smape), 4)})",
        'kernel_size': args.mask_kernel_size,
        "dropout": args.mask_dropout,
        'seeds': ' '.join(str(seed) for seed in args.seeds),
    }])], ignore_index=True)
    result_table.to_csv(result_path, index=False)
