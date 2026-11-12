"""Check the installation without any data: one forward/backward pass of both task configurations
on random input, plus the parameter count.

    python scripts/smoke_test.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asam.config import get_config  # noqa: E402
from asam.model import ASAM  # noqa: E402

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    sys.exit('mamba-ssm requires a CUDA GPU')

for task in ('adni_cn_mci', 'hbn_age'):
    cfg = get_config(task).model
    model = ASAM(cfg).to(device)
    left = torch.randn(2, cfg.n_patches, cfg.n_vertices, cfg.in_channels, device=device)
    right = torch.randn_like(left)
    out = model(left, right)
    out.sum().backward()
    n = sum(p.numel() for p in model.parameters())
    print(f'{task:12s} output {tuple(out.shape)}  parameters {n / 1e6:.2f}M  OK')
