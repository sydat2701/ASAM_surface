"""Configuration of ASAM.

The defaults below are the final configuration reported in the paper
(Sec. 3.3, "Implementation details"):

    embedding dimension D = 384, Mamba state size N = 16,
    K_min = 1, K_max = 5, kappa = 0.6, epsilon = 0.8, alpha = 0.5.

Two task presets are provided, one per main experiment:
    ``adni_cn_mci`` - CN vs. MCI classification on ADNI-1
    ``hbn_age``     - brain age regression on HBN
They differ only in the depth of the Attention branch and in the training
protocol (loss, epochs, sampler, ...), exactly as used for the released weights.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import Optional


@dataclass
class ModelConfig:
    # ---- input -------------------------------------------------------------
    in_channels: int = 3          # curvature, sulcal depth, cortical thickness
    n_patches: int = 320          # patches per hemisphere (Ico-6 -> 320 patches)
    n_vertices: int = 153         # vertices per patch

    # ---- backbone ----------------------------------------------------------
    embed_dim: int = 384          # D
    mamba_d_state: int = 16       # Mamba state size N
    n_heads: int = 8              # attention heads
    mamba_depth: int = 1          # number of Mamba blocks
    attention_depth: int = 6      # number of Attention blocks
    use_channel_attention: bool = True
    ca_depth: int = 6             # number of stacked channel-attention (CA) layers
    drop_rate: float = 0.4

    # ---- local feature learning (DMHS / DFM) -------------------------------
    k_min: int = 1                # K_min, smallest receptive field (hops)
    k_max: int = 5                # K_max, largest candidate receptive field (hops)
    kappa: float = 0.6            # survival-probability threshold of the hop expansion
    epsilon: float = 0.8          # cut-off on the fused affinity A^0
    alpha: float = 0.5            # weight of geodesic vs. dynamic affinity

    # ---- regularisation ----------------------------------------------------
    train_noise_std: float = 1.0  # Gaussian noise added to the input patches during training

    # ---- head --------------------------------------------------------------
    n_outputs: int = 1            # sigmoid output: P(MCI) for classification, age/label_scale for regression


@dataclass
class TrainConfig:
    task_type: str = 'classification'   # 'classification' | 'regression'
    epochs: int = 120
    batch_size: int = 8
    eval_batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_power: float = 0.9               # polynomial decay  lr_t = lambda * lr_0 * (1 - t/T)^gamma
    lr_scale: float = 0.8               # lambda
    balanced_sampler: bool = True       # uniform over-sampling of the minority class
    early_stop_patience: Optional[int] = None
    label_scale: float = 1.0            # regression targets are divided by this value
    per_patch_znorm: bool = False       # z-normalise every patch over its vertices
    seed: int = 55
    seed_every_fold: bool = True
    n_folds: int = 5


@dataclass
class Config:
    task: str = 'adni_cn_mci'
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def get_config(task: str) -> Config:
    """Return the configuration used for a task of the paper."""
    if task == 'adni_cn_mci':
        return Config(
            task=task,
            model=ModelConfig(attention_depth=6),
            train=TrainConfig(
                task_type='classification',
                epochs=120,
                batch_size=8,
                eval_batch_size=1,
                balanced_sampler=True,
                per_patch_znorm=False,   # ADNI arrays are already z-normalised per channel
                seed=55,
                seed_every_fold=True,
            ),
        )
    if task == 'hbn_age':
        return Config(
            task=task,
            model=ModelConfig(attention_depth=5),
            train=TrainConfig(
                task_type='regression',
                epochs=250,
                batch_size=24,
                eval_batch_size=32,
                balanced_sampler=False,
                early_stop_patience=60,
                label_scale=100.0,       # ages are learned in units of 100 years
                per_patch_znorm=True,
                seed=55,
                seed_every_fold=False,
            ),
        )
    raise ValueError(f"unknown task '{task}', expected one of: adni_cn_mci, hbn_age")


TASKS = ('adni_cn_mci', 'hbn_age')


def override(cfg: Config, **kwargs) -> Config:
    """Override fields of the model / train sub-configs by name (None values are ignored)."""
    model_kw = {k: v for k, v in kwargs.items() if v is not None and k in ModelConfig.__dataclass_fields__}
    train_kw = {k: v for k, v in kwargs.items() if v is not None and k in TrainConfig.__dataclass_fields__}
    return replace(cfg, model=replace(cfg.model, **model_kw), train=replace(cfg.train, **train_kw))
