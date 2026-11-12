"""Evaluate ASAM checkpoints on the held-out fold of each checkpoint and report mean (SD) over folds.

Each ``fold{k}.pth`` in ``--weights_dir`` is evaluated on the test subjects of fold k.

Examples
--------
    python evaluate.py --task adni_cn_mci --data_dir data/adni --weights_dir weights/adni_cn_mci
    python evaluate.py --task hbn_age     --data_dir data/hbn  --weights_dir weights/hbn_age
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from asam.config import TASKS, get_config
from asam.data import load_dataset, make_folds
from asam.engine import make_loaders, predict, compute_metrics
from asam.model import ASAM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--weights_dir', required=True, help='folder with fold0.pth ... fold4.pth')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--split_file', default=None)
    p.add_argument('--save_predictions', default=None, help='optional CSV with per-subject predictions')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.task)
    X, y, subs = load_dataset(args.task, args.data_dir, cfg.train.per_patch_znorm)
    folds = make_folds(args.task, y, subs, args.split_file, cfg.train.n_folds)

    rows, preds = [], []
    for k in args.folds:
        _, test_idx = folds[k]
        model = ASAM(cfg.model)
        state = torch.load(os.path.join(args.weights_dir, f'fold{k}.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(args.device)
        _, loader = make_loaders(X, y, test_idx, test_idx, cfg.train)
        y_true, out = predict(model, loader, args.device)
        m = compute_metrics(cfg.train.task_type, y_true, out, cfg.train.label_scale)
        rows.append({'fold': k, 'n_test': len(test_idx), **m})
        print(f'fold {k}: ' + ', '.join(f'{n}={v:.4f}' for n, v in m.items()))
        scale = cfg.train.label_scale if cfg.train.task_type == 'regression' else 1.0
        preds.append(pd.DataFrame({'subject_id': subs[test_idx], 'fold': k,
                                   'target': y_true * scale, 'output': out * scale}))

    table = pd.DataFrame(rows)
    metrics = [c for c in table.columns if c not in ('fold', 'n_test')]
    pct = 100.0 if cfg.train.task_type == 'classification' else 1.0
    print('\nmean (SD) over folds, SD with ddof=0 as in the paper' + (' [%]' if pct == 100 else ''))
    for c in metrics:
        v = table[c].values * pct
        print(f'  {c:>4}: {v.mean():.2f} ({v.std(ddof=0):.2f})')
    if args.save_predictions:
        pd.concat(preds).to_csv(args.save_predictions, index=False)
        print(f'predictions written to {args.save_predictions}')


if __name__ == '__main__':
    main()
