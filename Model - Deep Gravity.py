import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deep_gravity_utils as dgu
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset


# Model specification
class DeepGravityNet(nn.Module):
    def __init__(self, inp_dim, hid_dim=256, out_dim=1, dropout_p=0.35):
        super(DeepGravityNet, self).__init__()
        
        self.deep_graivty_fc1 = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p)
        )
        
        self.deep_graivty_fc2 = nn.Sequential(
            
            nn.Linear(hid_dim, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hid_dim // 2, inp_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p)
        )
        self.out_fc = nn.Sequential(
            nn.Linear(inp_dim, 1),
            nn.ReLU()
        )
    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = nn.BatchNorm1d(x.shape[1], device=device)(x)
        y = self.deep_graivty_fc1(x)
        y = nn.BatchNorm1d(y.shape[1], device=device)(y)
        # shape of "x" here = [batchsize, sequence_len, 256]
        y = self.deep_graivty_fc2(y)
        y += x
        y = self.out_fc(y)
        return y


def main():
    def get_data_X(dest):
        data_X_df = data_X_org.copy()
        # dest means the destination station name
        dest_data = data_X_df.loc[data_X_df['nearest station']==dest]
        dest_lat = data_X_df.loc[data_X_df['nearest station']==dest, 'INTPTLAT'].values
        dest_lon = data_X_df.loc[data_X_df['nearest station']==dest, 'INTPTLON'].values
        
        dist_lst = []
        for i in data_X_df.index:
            org_lat = data_X_df.loc[i, 'INTPTLAT']
            org_lon = data_X_df.loc[i, 'INTPTLON']
            dist = np.sqrt((dest_lat-org_lat)**2 + (dest_lon-org_lon)**2) * 111  # km
            dist_lst.append(dist[0])
        
        data_X_df.insert(2, 'dist', dist_lst)
        
        data_X_df = data_X_df.drop(['INTPTLAT', 'INTPTLON', 'nearest station'], axis=1)
        data_X = data_X_df.to_numpy()
        
        dest_data = dest_data.drop(['INTPTLAT', 'INTPTLON', 'nearest station'], axis=1)
        dest_data = dest_data.to_numpy().reshape(1, -1)
        dest_data_mat = np.repeat(dest_data, data_X.shape[0], axis=0)
        
        # concatenate data_X and dest_data_mat
        data_X = np.concatenate([data_X, dest_data_mat], axis=1)
            
        return data_X

    def get_data_X_for_all_stops(stops):
        # stops should be a list

        data_X = get_data_X(stops[0])
        data_y = OD[:, 0].reshape(-1, 1)
        data_y = (data_y - data_y.min()) / (data_y.max() - data_y.min())

        for i in range(1, len(stops)):
            stop = stops[i]
            data_X = np.concatenate([data_X, get_data_X(stop)], axis=0)
            
            y = OD[:, i].reshape(-1, 1)
            y = (y - y.min()) / (y.max() - y.min())
            data_y = np.concatenate([data_y, y], axis=0)
        
        # normalize
        data_X = (data_X - data_X.min()) / (data_X.max() - data_X.min())
    #     data_y = (data_y - data_y.min()) / (data_y.max() - data_y.min())
        
        return data_X, data_y

    # Read data
    OD = np.load('./data/3d_daily.npy').sum(axis=2)[:48, :48]
    data_X_org = pd.read_csv('./data/data_X.csv', index_col=0).iloc[:48, :]
    stops = pd.read_csv('./data/stops_order.csv', index_col=0).iloc[:48, :]

    # annual data
    data_X, data_y = get_data_X_for_all_stops(list(stops['stop']))

    inp_dim = data_X.shape[-1]

    # prepare data for pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = 0.7
    l = int(data_X.shape[0]*prop)
    X_train = torch.tensor(data_X[:l, :], dtype=torch.float32).to(device)
    y_train = torch.tensor(data_y[:l, :], dtype=torch.float32).to(device)
    X_val = torch.tensor(data_X[l:, :], dtype=torch.float32).to(device)
    y_val = torch.tensor(data_y[l:, :], dtype=torch.float32).to(device)

    batch_size = 2500
    loader_train = torch.utils.data.DataLoader(
        TensorDataset(X_train, y_train), batch_size, shuffle=True
    )
    iter_train = iter(loader_train)

    # initialize model
    model = DeepGravityNet(inp_dim).to(device)
    loss_func = nn.MSELoss()
    loss_set_train = []
    loss_set_val = []
    optimizer = optim.Adam(model.parameters())

    # train
    model.train()

    epochs = 1000

    # assert len(src_loader) == len(tar_loader)
    
    for e in range(epochs):
        #ipdb.set_trace()
        for i in range(len(loader_train)):
            try:
                X, y = iter_train.next()
            except:
                iter_train = iter(loader_train)
                X, y = iter_train.next()
            
            out = model(X)
            loss = loss_func(out, y)
            out_val = model(X_val)
            loss_val = loss_func(out_val, y_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_set_val.append(loss_val.cpu().detach().numpy())
            loss_set_train.append(loss.cpu().detach().numpy())

    def plot_loss(val, train, title):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(val)
        ax1.set_xlabel('Batches')
        ax1.set_ylabel('MSELoss')
        ax1.set_ylim([0, max(val)])
        ax1.set_title('Validation loss')
        ax1.grid()
        ax2 = fig.add_subplot(122)
        ax2.plot(train)
        ax2.set_xlabel('Batches')
        ax2.set_ylabel('MSELoss')
        ax2.set_ylim([0, max(train)])
        ax2.set_title('Train loss')
        ax2.grid()
        plt.suptitle(title)
        return fig

    # plot loss
    fig_loss = plot_loss(loss_set_val, loss_set_train, 'Loss')

    # torch.save(model.state_dict(), './models/model_48stations.pth')

    pred = out_val.cpu().detach().numpy()
    labels = y_val.cpu().detach().numpy()

    # plot result
    pred_df = pd.DataFrame(pred).sort_values(by=0, ascending=False)
    pred_df.index = range(pred_df.shape[0])
    labels_df = pd.DataFrame(labels).sort_values(by=0, ascending=False)
    labels_df.index = range(labels_df.shape[0])
    fig_res = plt.figure(figsize=[20,5], dpi=300)
    ax0 = fig_res.add_subplot(1, 1, 1)
    ax0.plot(pred_df[0], '.', label='pred', ms=10)
    ax0.plot(labels_df[0], '.', label='ground truth', ms=10)
    ax0.set_xlabel('Stations')
    ax0.set_ylabel('# Trips')
    plt.legend()
    plt.grid(ls='--')

    m = 0.0
    mae = dgu.mae_loss_func(pred, labels, m)
    mape = dgu.mape_loss_func(pred, labels, m)
    smape = dgu.smape_loss_func(pred, labels, m)
    nrmse = dgu.nrmse_loss_func(pred, labels, m)
    nmae = dgu.nmae_loss_func(pred, labels, m)

    result_df = pd.DataFrame(
        np.array([[mae, mape, smape, nrmse, nmae]]),\
        index=[0], columns=['mae', 'mape', 'smape', 'nrmse', 'nmae']
        )

    num_fold = dgu.get_num_fold()
    os.mkdir('./runs/run%i'%(num_fold+1))

    dgu.save_model(model, 'last')
    dgu.save_fig(fig_loss, 'fig_loss')
    dgu.save_fig(fig_res, 'fig_res')
    dgu.save_res(result_df, 'result_df')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute dg model numtiple times')
    parser.add_argument('--iteration', type=int, help='Number of iterations', required=True)
    args = parser.parse_args()
    n_iter = args.iteration

    for i in range(n_iter):
        print('Start iteration #%i'%i)
        main()