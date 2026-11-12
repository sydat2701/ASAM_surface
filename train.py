"""5-fold cross-validation training of ASAM.

Examples
--------
ADNI CN vs. MCI classification:
    python train.py --task adni_cn_mci --data_dir data/adni --out_dir runs/adni_cn_mci
HBN brain age prediction:
    python train.py --task hbn_age --data_dir data/hbn --out_dir runs/hbn_age
"""
import argparse
import json
import os

import torch

from asam.config import TASKS, get_config, override
from asam.data import load_dataset, make_folds
from asam.engine import seed_everything, make_loaders, train_fold
from asam.model import ASAM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--data_dir', required=True, help='folder with data_ico6.npy, labels.npy, subs.npy')
    p.add_argument('--out_dir', required=True, help='checkpoints and logs are written here')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--split_file', default=None,
                   help="fold assignment CSV (default: the released splits/*.csv); 'none' = regenerate")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # optional overrides of the paper configuration
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--k_max', type=int)
    p.add_argument('--kappa', type=float)
    p.add_argument('--epsilon', type=float)
    p.add_argument('--alpha', type=float)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = override(get_config(args.task), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   seed=args.seed, k_max=args.k_max, kappa=args.kappa, epsilon=args.epsilon, alpha=args.alpha)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    X, y, subs = load_dataset(args.task, args.data_dir, cfg.train.per_patch_znorm)
    folds = make_folds(args.task, y, subs, args.split_file, cfg.train.n_folds)
    print(f'{args.task}: {len(y)} subjects, data {X.shape}')

    if not cfg.train.seed_every_fold:
        seed_everything(cfg.train.seed)
    for k in args.folds:
        if cfg.train.seed_every_fold:
            seed_everything(cfg.train.seed)
        train_idx, test_idx = folds[k]
        model = ASAM(cfg.model)
        if k == args.folds[0]:
            print(f'ASAM parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
        print(f'===== fold {k}: train {len(train_idx)}, test {len(test_idx)} =====')
        train_loader, test_loader = make_loaders(X, y, train_idx, test_idx, cfg.train)
        ckpt = train_fold(model, train_loader, test_loader, cfg.train, args.device, args.out_dir, k)
        print(f'best checkpoint of fold {k}: {ckpt}')

    print(f'done. Evaluate with:\n  python evaluate.py --task {args.task} --data_dir {args.data_dir} '
          f'--weights_dir {args.out_dir}')


if __name__ == '__main__':
    main()
