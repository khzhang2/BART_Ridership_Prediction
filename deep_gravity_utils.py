import numpy as np
import pandas as pd
import os
import random

import torch
from torch import nn


def mape_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(eliminate_nan(np.fabs(labels[mask]-preds[mask])/labels[mask]))


def smape_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))


def mae_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(np.fabs((labels[mask]-preds[mask])))


def nrmse_loss_func(preds, labels, m):
    mask = preds > m
    return np.sqrt(np.sum((preds[mask] - labels[mask])**2)/preds[mask].flatten().shape[0])/(labels[mask].max() - labels[mask].min())


def nmae_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(np.fabs((labels[mask]-preds[mask]))) / (labels[mask].max() - labels[mask].min())


def eliminate_nan(b):
    a = np.array(b)
    c = a[~np.isnan(a)]
    c = c[~np.isinf(c)]
    return c


def get_class(v):
    # v is 1-d or 2-d array
    # we set that there are 100 classes between 0 and 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(v.shape) == 1:
        try:
            v = v.reshape(-1, v.shape[0])
        except:
            v = v.view(-1, v.shape[0])

    try:
        v = np.array(v)
        v_cls = np.zeros_like(v)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_cls[i, j] = int(np.floor(v[i, j]*100))//1
        return torch.tensor(v_cls, dtype=torch.float32).to(device)
    except:
        None
        # v_cls = torch.zeros_like(v)
        # for i in range(v.shape[0]):
        #     for j in range(v.shape[1]):
        #         v_cls[i, j] = int(torch.floor(v[i, j]*100))//1
        # return torch.tensor(v_cls, dtype=torch.float32).to(device)


def normalize2D_tSNE(V):
    V = np.array(V)
    return ( V ) / ( V.max(0) - V.min(0) ), V.min(0), V.max(0)


def normalize2D(V):
    V = np.array(V)
    return ( V - V.min(0) ) / ( V.max(0) - V.min(0) ), V.min(0), V.max(0)


def denormalize2D(V, V_min, V_max):
    V = np.array(V)
    V_min = np.array(V_min)
    V_max = np.array(V_max)
    denormalized_V = V * (V_max - V_min) + V_min
    return denormalized_V


def sliding_window(T, T_org, seq_len, label_seq_len):  # was: (T, T_org, seq_len, label_seq_len)
    # seq_len is equal to window_size
    # T (np.ndarray) has dim: sample size, dim
    K = T.shape[0] - seq_len - label_seq_len + 1  # Li, et al., 2021, TRJ part C, pp. 8
    
#     TT_org = T_org.reshape(-1, 1)

    # assemble the data into 3D
    x_set = T[:K, np.newaxis, :]
#     x_set = np.concatenate(TT[i : K+i, 0] for i in range(seq_len), axis=1)
    for i in range(1, seq_len):
        x_set = np.concatenate((x_set, T[i:K+i, np.newaxis, :]), axis=1)
    
    y_set = T_org[seq_len:K+seq_len, np.newaxis, :]
    for i in range(1, label_seq_len):
        y_set = np.concatenate((y_set, T_org[i+seq_len:K+i+seq_len, np.newaxis, :]), axis=1)
    
#     y_set = np.vstack(T_org[i+seq_len : K+seq_len+i, 0] for i in range(label_seq_len)).T
    
    assert x_set.shape[0] == y_set.shape[0]

    # return size: n_samp, seq_len
    return x_set, y_set


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def save_model(net, name):
    num_fold = get_num_fold() + 1
    try:
        torch.save(net.state_dict(), './runs/run%i/%s.pth'%(num_fold, name))
    except:
        raise RuntimeError('No fold for this experiment created')

def get_num_fold():
    num_fold = len(next(iter(os.walk('./runs/')))[1])
    return num_fold
