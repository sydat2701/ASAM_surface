"""Build the ASAM input arrays (data_ico6.npy, labels.npy, subs.npy) from per-subject Ico-6 metric files.

Input layout (output of the preprocessing pipeline, https://github.com/sydat2701/brain_surface_preprocessing):
    <surf_dir>/<subject_id>/{lh,rh}.{curv,sulc,thickness}.fsaverage.shape.gii
i.e. FreeSurfer curvature, sulcal depth and thickness mapped to fsaverage and resampled to the
Ico-6 sphere (40,962 vertices per hemisphere), e.g. with Connectome Workbench:
    wb_command -metric-resample lh.thickness.fsaverage.gii resources/ico6/lh.average.surf.gii \
        resources/ico6/ico-6.L.surf.gii BARYCENTRIC lh.thickness.fsaverage.shape.gii

<labels_csv> has the columns ``subject_id,label`` (ADNI: CN / sMCI / pMCI / Dementia; HBN: age in years).

Each hemisphere is cut into the 320 non-overlapping triangular patches of
resources/ico6/patches_ico6_320patches.csv (153 vertices each); patches 0-319 are the left hemisphere,
patches 320-639 the right hemisphere.

Normalisation used in the paper:
    ADNI : --channel_znorm   (z-score of every channel over all subjects, zeros of the medial wall excluded)
    HBN  : raw values; train.py applies a per-subject, per-patch z-normalisation at load time.

Examples:
    python scripts/make_patch_arrays.py --surf_dir /data/adni_ico6 --labels_csv adni.csv --out_dir data/adni --channel_znorm
    python scripts/make_patch_arrays.py --surf_dir /data/hbn_ico6  --labels_csv hbn.csv  --out_dir data/hbn
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ('curv', 'sulc', 'thickness')
ADNI_CLASSES = {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'Dementia': 3, 'AD': 3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--surf_dir', required=True)
    p.add_argument('--labels_csv', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--channel_znorm', action='store_true')
    p.add_argument('--patches_csv', default=os.path.join(REPO_DIR, 'resources', 'ico6', 'patches_ico6_320patches.csv'))
    args = p.parse_args()

    table = pd.read_csv(args.labels_csv, dtype={'subject_id': str})
    patches = pd.read_csv(args.patches_csv)                               # (153 vertices, 320 patches)
    idx = np.stack([patches[str(j)].values for j in range(patches.shape[1])])  # (320, 153)

    subs, labels, hemis = [], [], []
    for sid, lab in zip(table.subject_id, table.label):
        files = [os.path.join(args.surf_dir, sid, f'{h}.{f}.fsaverage.shape.gii') for h in ('lh', 'rh') for f in FEATURES]
        if not all(os.path.exists(f) for f in files):
            print(f'skipping {sid}: missing metric files')
            continue
        data = np.stack([nib.load(f).agg_data() for f in files]).reshape(2, len(FEATURES), -1)  # (hemi, C, 40962)
        hemis.append(data)
        subs.append(sid)
        labels.append(ADNI_CLASSES.get(lab, lab) if isinstance(lab, str) else lab)
    hemis = np.stack(hemis).astype(np.float64)                             # (N, 2, C, 40962)

    if args.channel_znorm:
        for c in range(hemis.shape[2]):
            v = hemis[:, :, c]
            nz = v[v != 0]
            hemis[:, :, c] = (v - nz.mean()) / nz.std()

    X = np.concatenate([hemis[:, 0][:, :, idx], hemis[:, 1][:, :, idx]], axis=2)  # (N, C, 640, 153)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'data_ico6.npy'), X)
    np.save(os.path.join(args.out_dir, 'labels.npy'), np.asarray(labels))
    np.save(os.path.join(args.out_dir, 'subs.npy'), np.asarray(subs))
    print(f'saved {X.shape} to {args.out_dir}')


if __name__ == '__main__':
    main()
