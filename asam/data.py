"""Data loading, normalisation and cross-validation splits.

Expected input (one folder per dataset, produced by the preprocessing pipeline):
    data_ico6.npy  float, (N, 3, 640, 153)  channels = [curvature, sulcal depth, thickness];
                   patches 0-319 = left hemisphere, 320-639 = right hemisphere;
                   153 Ico-6 vertices per patch (resources/ico6/patches_ico6_320patches.csv)
    labels.npy     (N,)  ADNI: 0 = CN, 1 = sMCI, 2 = pMCI, 3 = AD;  HBN: age in years
    subs.npy       (N,)  subject ids (used to match the released fold files)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_FILES = {
    'adni_cn_mci': os.path.join(REPO_DIR, 'splits', 'adni_cn_mci_5fold.csv'),
    'hbn_age': os.path.join(REPO_DIR, 'splits', 'hbn_age_5fold.csv'),
}


def z_normalize_per_patch(x, eps=1e-8):
    """Z-normalise every (subject, channel, patch) over its vertices."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def load_dataset(task, data_dir, per_patch_znorm=False):
    """Return X (N, 3, 640, 153) float32, y (N,), subject ids (N,) for a task."""
    X = np.load(os.path.join(data_dir, 'data_ico6.npy')).astype('float32')
    y = np.load(os.path.join(data_dir, 'labels.npy'))
    subs = np.load(os.path.join(data_dir, 'subs.npy'), allow_pickle=True).astype(str)
    assert X.ndim == 4 and X.shape[1:] == (3, 640, 153), f'unexpected data shape {X.shape}'
    assert len(X) == len(y) == len(subs)

    if task == 'adni_cn_mci':
        keep = np.isin(y, [0, 1, 2])                     # CN vs. MCI (sMCI + pMCI)
        X, y, subs = X[keep], (y[keep] > 0).astype(np.int64), subs[keep]
    elif task == 'hbn_age':
        y = y.astype(np.float32)
    else:
        raise ValueError(task)

    if per_patch_znorm:
        X = z_normalize_per_patch(X)
    return X, y, subs


def make_folds(task, y, subs, split_file=None, n_folds=5):
    """List of (train_idx, test_idx) for the 5 folds.

    By default the released fold assignment (``splits/*.csv``, one row per subject) is used, so that
    the folds are identical to the paper regardless of the order of the subjects in the arrays.
    Pass ``split_file='none'`` to regenerate the folds with the original scikit-learn splitter instead
    (identical to the released files when the arrays have the same subjects in the same order).
    """
    split_file = SPLIT_FILES[task] if split_file is None else split_file
    if split_file != 'none':
        table = pd.read_csv(split_file, dtype={'subject_id': str})
        fold_of = dict(zip(table.subject_id, table.fold))
        missing = [s for s in subs if s not in fold_of]
        if missing:
            raise ValueError(f'{len(missing)} subjects are not in {split_file} (e.g. {missing[:3]}); '
                             f"use --split_file none to create new folds")
        fold = np.array([fold_of[s] for s in subs])
        return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_folds)]

    if task == 'adni_cn_mci':
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2022)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    return list(splitter.split(np.zeros(len(y)), y))


class SurfaceDataset(Dataset):
    """Returns ((left, right), label) with left/right of shape (320, 153, 3)."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(1, 2, 0)  # (640, 153, 3)
        L = x.shape[0] // 2
        return (x[:L], x[L:]), torch.tensor([self.y[i]], dtype=torch.float32)
