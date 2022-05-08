import numpy as np
import os
import random
import torch


def mape_loss_func(preds, labels, m):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = preds > m
    return np.mean(eliminate_nan(np.fabs(labels[mask]-preds[mask])/labels[mask]))


def smape_loss_func(preds, labels, m):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = preds > m
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))


def mae_loss_func(preds, labels, m):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = preds > m
    return np.mean(np.fabs((labels[mask]-preds[mask])))


def nrmse_loss_func(preds, labels, m):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = preds > m
    return np.sqrt(np.sum((preds[mask] - labels[mask])**2)/preds[mask].flatten().shape[0])/(labels[mask].max() - labels[mask].min())


def nmae_loss_func(preds, labels, m):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = preds > m
    return np.mean(np.fabs((labels[mask]-preds[mask]))) / (labels[mask].max() - labels[mask].min())


def eliminate_nan(b):
    a = np.array(b)
    c = a[~np.isnan(a)]
    c = c[~np.isinf(c)]
    return c


def get_CPC(pred, labels):
    pred = pred.reshape(-1, 1)
    labels = labels.reshape(-1, 1)
    res_min = np.concatenate([pred, labels], axis=1).min(axis=1).flatten()
    CPC = np.sum(res_min)*2 / (np.sum(pred) + np.sum(labels))
    return CPC


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_num_fold():
    num_fold = len(next(iter(os.walk('./runs/')))[1])
    return num_fold


def save_model(net, name):
    num_fold = get_num_fold()
    try:
        torch.save(net.state_dict(), './runs/run%i/%s.pth' % (num_fold, name))
    except Exception as e:
        raise RuntimeError('No fold for this experiment created:'+str(e))


def save_res(res, name):
    num_fold = get_num_fold()
    try:
        res.to_csv('./runs/run%i/%s.csv' % (num_fold, name))
    except Exception as e:
        raise RuntimeError('No fold for this experiment created:'+str(e))


def save_fig(fig, name):
    num_fold = get_num_fold()
    try:
        fig.savefig('./runs/run%i/%s.png' % (num_fold, name), dpi=300)
    except Exception as e:
        raise RuntimeError('No fold for this experiment created:'+str(e))


def const_4d_OD(OD, t_past, t_future):
    # input OD shape: [num_stations, num_stations, time_seq_len]
    OD = np.transpose(OD, (2, 0, 1))[:, np.newaxis, :, :]
    OD_4d = np.zeros(OD.shape)
    OD_4d = np.repeat(OD_4d, t_past+t_future, axis=1)  # past t_past days plus future t_future days
    for i in range(t_past + t_future, OD.shape[0]):
        OD_4d[i, :, :, :] = OD[i-t_past-t_future:i, 0, :, :]
        
    print('Memory occupied %.4f MB' % ((OD_4d.size * OD_4d.itemsize)/1024**2) )

    return OD_4d[t_past + t_future:, :, :, :]
