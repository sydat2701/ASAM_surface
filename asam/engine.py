"""Training / evaluation loops and metrics."""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import SurfaceDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def poly_lr(optimizer, epoch, total_epochs, lr0, power, scale):
    """lr_t = lambda * lr_0 * (1 - t / T)^gamma, updated once per epoch."""
    lr = round(lr0 * (1 - epoch / total_epochs) ** power, 8) * scale
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


# ------------------------------------------------------------------------------ metrics
def classification_metrics(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob > 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        'acc': (tp + tn) / len(y_true),
        'bacc': (sen + spe) / 2,
        'auc': roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float('nan'),
        'sen': sen,                                 # recall of MCI
        'spe': spe,
        'f1': tp / (tp + (fp + fn) / 2) if tp + fp + fn > 0 else 0.0,
        'pre': tp / (tp + fp) if tp + fp > 0 else 0.0,
    }


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return {'mae': float(np.mean(np.abs(y_true - y_pred))), 'r': float(pearsonr(y_pred, y_true)[0])}


def compute_metrics(task_type, y_true, out, label_scale=1.0):
    if task_type == 'classification':
        return classification_metrics(y_true, out)
    return regression_metrics(np.asarray(y_true) * label_scale, np.asarray(out) * label_scale)


# ------------------------------------------------------------------------------ loops
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outs, labels = [], []
    for (left, right), y in loader:
        out = model(left.to(device).float(), right.to(device).float())
        outs.append(out.view(-1).cpu().numpy())
        labels.append(y.view(-1).numpy())
    return np.concatenate(labels), np.concatenate(outs)


def make_loaders(X, y, train_idx, test_idx, tcfg):
    """Train / held-out loaders. Regression targets are divided by ``label_scale``."""
    y_t = y / tcfg.label_scale if tcfg.task_type == 'regression' else y
    train_set = SurfaceDataset(X[train_idx], y_t[train_idx])
    test_set = SurfaceDataset(X[test_idx], y_t[test_idx])
    if tcfg.balanced_sampler:
        y_tr = torch.as_tensor(y_t[train_idx]).long()
        weights = (1.0 / torch.bincount(y_tr).float())[y_tr]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=tcfg.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=tcfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=tcfg.eval_batch_size, shuffle=False)
    return train_loader, test_loader


def train_fold(model, train_loader, test_loader, tcfg, device, out_dir, fold):
    """Train one cross-validation fold and keep the checkpoint with the best held-out score
    (highest BACC for classification, lowest MAE for regression)."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    classification = tcfg.task_type == 'classification'
    criterion = nn.BCELoss() if classification else nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    ckpt = os.path.join(out_dir, f'fold{fold}.pth')
    log = open(os.path.join(out_dir, f'log_fold{fold}.jsonl'), 'w')

    best, best_epoch = None, -1
    for epoch in range(tcfg.epochs):
        lr = poly_lr(optimizer, epoch, tcfg.epochs, tcfg.lr, tcfg.lr_power, tcfg.lr_scale)
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f'fold {fold} epoch {epoch + 1}/{tcfg.epochs}', leave=False,
                    disable=not sys.stderr.isatty())
        for (left, right), y in pbar:
            optimizer.zero_grad()
            out = model(left.to(device).float(), right.to(device).float())
            loss = criterion(out, y.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        y_true, out = predict(model, test_loader, device)
        m = compute_metrics(tcfg.task_type, y_true, out, tcfg.label_scale)
        score = m['bacc'] if classification else -m['mae']
        improved = best is None or score > best
        if improved:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), ckpt)
        rec = {'epoch': epoch + 1, 'lr': lr, 'train_loss': running / len(train_loader), **m, 'best': improved}
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(f'fold {fold} ' + ' '.join(f'{k}={v:.3g}' if k == 'lr' else f'{k}={v:.4f}' if isinstance(v, float)
                                         else f'{k}={v}' for k, v in rec.items()), flush=True)

        if tcfg.early_stop_patience is not None and epoch - best_epoch >= tcfg.early_stop_patience:
            print(f'early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1})')
            break
    log.close()
    return ckpt
