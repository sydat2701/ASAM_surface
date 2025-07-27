
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
import gc
import random
import os
from utils.utils import seed_everything, train, get_task_data, test
import argparse
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error
from data.dataset import Feeder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import shutil
from torch.utils.data import WeightedRandomSampler
from model.model import ASAM
import configs.config as cf 

def z_normalize_per_patch(x, eps=1e-8, dataset_name = '', mean=None, std=None, self_calc = True):
    """
    Z-normalize over the V (vertices) dimension, per (B, C, P) unit.

    Args:
        x (np.ndarray): Input array of shape (B, C, P, V)
        eps (float): Small constant for numerical stability

    Returns:
        np.ndarray: Z-normalized array
    """
    if self_calc:
        mean = np.mean(x, axis=-1, keepdims=True)  # (B, C, P, 1)
        std = np.std(x, axis=-1, keepdims=True)    # (B, C, P, 1)
    np.save(f"mean_{dataset_name}.npy", mean)
    np.save(f"std_{dataset_name}.npy", std)
    return (x - mean) / (std + eps)

if __name__ == '__main__':
    #X: (n, channels, vertices)
    print("Preparing data...")
    X, y = np.load('data_hbn_3sites_totrain/data_ico6.npy').astype('float32'), np.load('data_hbn_3sites_totrain/labels.npy')
    X = z_normalize_per_patch(X)
    
    print(X.shape, y.shape)
    #-------------------------------------------------------------------------------------------------------
    epoch = cf.NUM_EPOCH
    model_ = 'ASAM'

    batch_size = 24
    seeds = [55]
    is_cosinelr = False
    criterion = nn.L1Loss()
    lr = 0.0001

    for seed in seeds:
        seed_everything(seed)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log_path = './history/logs_HBN/jun18_{}_seed{}_noLFL'.format(model_, seed)
        weight_path = './history/weights_HBN/jun18_{}_seed{}_noLFL'.format(model_, seed)

        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        if os.path.isdir(weight_path):
            shutil.rmtree(weight_path)
        
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(weight_path, exist_ok=True)
        #-------------------------------------------------------------------------------------------------------

        print("log at: ", log_path, "and model is ", model_)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>> Each fold will be trained for {epoch} epoch(s) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Training data: {X.shape} {y.shape}" )

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold = 0
        BACC = []
        SEN  = []
        SPE  = []
        AUC_SCORE  = []
        fol_idx = 0
        for train_idx, test_idx in kf.split(X, y):
            if fol_idx not in [0,1,2,3,4]:
                fol_idx +=1
                continue

            print("Initializing model...")
            if model_ == 'ASAM':
                model = ASAM()

            print("Number of params: ", sum(p.numel() for p in model.parameters()))
            print("Complete initializing...")
            # exit()

            gc.collect()
            fold += 1

            fold_weight_path = weight_path
            fold_log_path = log_path

            os.makedirs(fold_log_path, exist_ok=True)
            os.makedirs(fold_weight_path, exist_ok=True)

            print(f"*************************************FOLD: {fold}**************************************************")

            Xtrain, ytrain = X[train_idx], np.expand_dims(y[train_idx], axis=-1)
            Xtest, ytest = X[test_idx], np.expand_dims(y[test_idx], axis=-1)

            #------------------------sampler--------------------------------
            # class_counts = torch.bincount(torch.tensor(ytrain).view(-1))
            # class_weights = 1. / class_counts
            # sample_weights = class_weights[torch.tensor(ytrain).view(-1)]

            # sampler = WeightedRandomSampler(
            #     weights=sample_weights,
            #     num_samples=len(sample_weights),
            #     replacement=True
            # )
            #---------------------------------------------------------------
            train_feeder = Feeder(Xtrain, ytrain) #data shape: (B, C, P, V) for data Q
            val_feeder = Feeder(Xtest, ytest)

            dataloader_train = DataLoader(dataset=train_feeder, batch_size=batch_size, shuffle= True)
            dataloader_val = DataLoader(dataset=val_feeder, batch_size=32, shuffle= False)

            train(model, epoch, device, criterion, fold_weight_path, lr, dataloader_train, \
                dataloader_val, fol_idx, fold_log_path, is_cosinelr)
            # test(model, device=device, criterion=criterion, weight_path="history/weights_adni/TMP_SiT_verified_seed55", dataloader_val=dataloader_val, fol_idx=fol_idx)

            fol_idx +=1
