# ASAM: Adaptive Surface Representation with Attention and Mamba

This repository contains the official PyTorch implementation of **ASAM**, a hybrid framework designed for cortical surface-based neuroimaging tasks. ASAM combines Mamba-based sequence modeling with attention-based spatial reasoning in a parallel design, enhanced with geometry-aware modules for adaptive local context integration.

## 🔗 Preprocessing Pipeline

To prepare cortical surface data for this repository, please follow our preprocessing pipeline hosted at:

👉 [https://github.com/sydat2701/brain_surface_preprocessing](https://github.com/sydat2701/brain_surface_preprocessing)

This includes:
- Surface reconstruction using FreeSurfer
- Resampling to icosphere templates
- Patch extraction and feature generation
- Outputting `*.gii` and `*.npy` inputs used by this repository
