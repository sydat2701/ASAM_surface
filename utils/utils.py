import os

import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from utils.eval_metrics import report
import random
import copy
import nibabel as nib
import scipy.stats
from sklearn.metrics import mean_absolute_error


from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9): 
    lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)*0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)

def cosine_scheduler(optimizer, initial_lr, min_lr, epochs_per_cycle, epoch):
    cycle = 50
    
    lr = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % cycle) / cycle)) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)


'''
    DATA UTILITY
'''

def get_task_data(X, y, task):
    neg_idx = pos_idx = -1
    X_return = []
    y_return = []
    if task == 'cn_ad':
        neg_idx = 0
        pos_idx = 3
    elif task == 'cn_mci':
        neg_idx = 0
        pos_idx = (1, 2)

    labels = np.zeros((4))
    for i, class_id in enumerate(y):
        labels[class_id] +=1

        if class_id == neg_idx:
            X_return.append(X[i])
            y_return.append(0)
        
        if isinstance(pos_idx, tuple):
            if class_id in pos_idx:
                X_return.append(X[i])
                y_return.append(1)
                
        elif class_id == pos_idx:
            X_return.append(X[i])
            y_return.append(1)
            
    return np.asarray(X_return), np.asarray(y_return)



def train(model, num_epoch, device, criterion, weight_path, lr, dataloader_train, dataloader_val, \
          fol_idx, log_path, is_cosinelr):
    
    global latest_saved_weight
    latest_saved_weight = None

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    if is_cosinelr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 1e-5)

    best_val_loss= 99999999
    best_val_acc = -99999999
    best_val_mae = 99999999
    num100 = 0
    history_train = np.zeros((num_epoch, 2)) #loss and auc
    history_val = np.zeros((num_epoch, 2))
    ep100 = None
    best_ep=None

    for epoch in range(num_epoch):
        print("Epoch: {}".format(epoch+1))
        epoch_train_acc = 0
        epoch_train_loss=0
        epoch_loss_genx = 0
        epoch_loss_geny = 0
        epoch_pearson_corr = 0
        num_iters_train=0
        model.train()
        
        adjust_learning_rate(optimizer, epoch, num_epoch, lr)
        pbar_train = tqdm(dataloader_train)
       
        for inputs, labels in pbar_train:
            optimizer.zero_grad()

            labels = labels.to(device).float()
            num_iters_train += 1
            y_pred = model([inputs[0].to(device).float()], [inputs[1].to(device).float()], epoch+1)


            loss= criterion(y_pred, labels)

            tmp1 =y_pred.view(-1).detach().cpu().numpy()
            tmp2 = labels.detach().cpu().view(-1).numpy()
            if np.std(tmp1) < 1e-8 and epoch<10: 
                tmp1 = tmp1 + np.random.normal(scale=1e-6, size=tmp1.shape)

            pcorr = scipy.stats.pearsonr(tmp1, tmp2)
            
            epoch_train_loss += loss
            epoch_pearson_corr += pcorr[0]

            loss.backward()
            optimizer.step()

            pbar_train.set_description(f"Loss: {epoch_train_loss.item()/num_iters_train:.4f}, pcorr: {epoch_pearson_corr/num_iters_train:.4f}")
            
        train_loss = epoch_train_loss / num_iters_train
        train_pear = epoch_pearson_corr/num_iters_train
        history_train[epoch, 0] =  train_loss.item()
        
      
        if is_cosinelr:
            sched.step()
            for param_group in optimizer.param_groups:
                print("Current lr: ", param_group['lr'])
        
        log_train = open(os.path.join(log_path, 'train.txt'), 'a')
        log_train.write('Epoch {}: train_loss {}, train_pear_corr\n'.format(epoch, \
                                                                       (train_loss).item(), train_pear))
        log_train.close()

        epoch_train_loss =0
        epoch_train_acc=0
        num_iters_train=0
        print("train_loss MAE:", train_loss.item())

        #---------------------------------------VALIDATION---------------------------
        epoch_val_acc = 0
        epoch_val_loss=0
        num_iters_val = 0
        epoch_val_pearson_corr = 0
        pearson_preds = []
        pearson_labs = []

        list_labels = []
        list_preds = []
        list_preds_prob = []

        model.eval()
        f = open('log_val_pred.txt', 'a')
        if epoch == 0:
            f.write("######################################{}############################################\n".format(fol_idx))
        for inputs, labels in tqdm(dataloader_val):
            labels = labels.to(device).float()

            num_iters_val += 1
            with torch.no_grad():
                y_pred = model([inputs[0].to(device).float()], [inputs[1].to(device).float()], epoch+1, mode='val')
                loss = criterion(y_pred, labels)
                tmp1 =y_pred.view(-1).detach().cpu().numpy()
            
                if np.std(tmp1) < 1e-8 and epoch<10:  # near-constant
                    tmp1 = tmp1 + np.random.normal(scale=1e-6, size=tmp1.shape)

                pearson_preds += tmp1.tolist()
                pearson_labs += labels.view(-1).cpu().numpy().tolist()

            epoch_val_loss += loss
            epoch_val_pearson_corr += pcorr[0]

            #-------------------------------------------------------------------           
            f.write(str(labels.cpu().detach().numpy())+'\n')
            f.write(str(y_pred.cpu().detach().numpy())+'\n')
            f.write("---------------------------------------------\n")

        val_pcorr = scipy.stats.pearsonr(np.array(pearson_preds), np.array(pearson_labs))[0]
        val_loss = mean_absolute_error(pearson_labs, pearson_preds)
        
        epoch_val_loss =0
        epoch_val_acc=0
        num_iters_val=0
        f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.close()
        print("val_loss:", val_loss.item(), ", val_pcorr: ", val_pcorr)

        log_val = open(os.path.join(log_path, 'val.txt'), 'a')
        log_val.write('Epoch {}: val_loss {}\n'.format(epoch, val_loss.item()))
        log_val.close()

        history_val[epoch, 0] = val_loss.item()

        plot_loss_and_metric(history_train, os.path.join(log_path, f'train._fol{fol_idx}.png'), epoch)
        plot_loss_and_metric(history_val, os.path.join(log_path, f'val_fol{fol_idx}.png'), epoch, mode='Val')

        if val_loss< best_val_mae:
            best_ep = epoch
            best_val_mae = val_loss

            if latest_saved_weight !=None:
                os.remove(latest_saved_weight)
            tmp_path = weight_path+'/'+ f'mae{val_loss*100}_pearcorr{val_pcorr}_fol'+str(fol_idx)+'.pth'
            latest_saved_weight = tmp_path
            torch.save(model.state_dict(), tmp_path)
            print("----------------------->saved model<--------------------")
            
            f = open(weight_path+f'/test_fol{fol_idx}.txt','w')
            f.write("Num test subs: " + str(len(pearson_preds))+'\n')
            f.write(f"Test MAE: {val_loss}, pcorr: {val_pcorr}"+'\n')
            f.write("PRED: " +str(pearson_preds)+'\n')
            f.write("LAB: " +str(pearson_labs)+'\n')

            f.close()

        if epoch-best_ep>=60:
            break
from glob import glob
def test(model, device, criterion, weight_path, dataloader_val, \
          fol_idx):
    
    global latest_saved_weight
    latest_saved_weight = None

    model.to(device)
    print(weight_path+f'/*_fol{fol_idx}.pth')
    w_path = glob(weight_path+f'/*_fol{fol_idx}.pth')[0]
    model.load_state_dict(torch.load(w_path, map_location=device), strict=False)
    print(f"Load successfully weight of fol {fol_idx+1}!")

    
    best_val_loss= 99999999
    best_val_acc = -99999999
    best_val_bacc = -99999999
    num100 = 0
    #---------------------------------------VALIDATION---------------------------
    epoch_val_acc = 0
    epoch_val_loss=0
    num_iters_val = 0

    pearson_preds = []
    pearson_labs = []
    feats = []


    model.eval()
    statistic_roi = torch.zeros((6))
    for inputs, labels in tqdm(dataloader_val):
        labels = labels.to(device).float()

        num_iters_val += 1
        with torch.no_grad():
            y_pred, num_roi_hop = model([inputs[0].to(device).float()], [inputs[1].to(device).float()], 0, mode='val')
            loss = criterion(y_pred, labels)
            tmp1 =y_pred.view(-1).detach().cpu().numpy()
            statistic_roi += num_roi_hop.cpu()

            pearson_preds += tmp1.tolist()
            pearson_labs += labels.view(-1).cpu().numpy().tolist()

        epoch_val_loss += loss

    epoch_val_loss =0
    epoch_val_acc=0
    num_iters_val=0
    val_pcorr = scipy.stats.pearsonr(np.array(pearson_preds), np.array(pearson_labs))[0]
    val_loss = mean_absolute_error(pearson_labs, pearson_preds)

    print("Test_loss:", val_loss.item(), ", Test_pcorr: ", val_pcorr)

    f = open(weight_path+f'/test_abide_fol{fol_idx}.txt','a')
    f.write("Num test subs: " + str(len(pearson_preds))+'\n')
    f.write(f"Test MAE: {val_loss}, pcorr: {val_pcorr}"+'\n')
    f.write("PRED: " +str(pearson_preds)+'\n')
    f.write("LAB: " +str(pearson_labs)+'\n')
    f.close()
    print("==================== roi: ", statistic_roi)


import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_metric(array, path_to_save_img, current_epoch, mode='Train'):
    """
    Plots loss and a given metric (e.g., AUC) up to the current epoch.
    
    Args:
        array (numpy.ndarray): Shape (num_epochs, 2), where first column is loss and second is the metric.
        path_to_save_img (str): Path to save the plot image.
        name_of_metric (str): Name of the metric (e.g., "AUC").
        current_epoch (int): The current epoch number.
    """
    # Ensure array is sliced correctly
    epochs = np.arange(1, current_epoch + 1)
    loss = array[:current_epoch, 0]
    
    plt.figure(figsize=(8, 6))
    
    # Plot loss
    plt.plot(epochs, loss, label='Loss', color='red', marker='o', linestyle='-')
    plt.savefig(path_to_save_img, dpi=300, bbox_inches='tight')
    plt.close()