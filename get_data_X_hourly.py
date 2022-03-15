import numpy as np
import pandas as pd

import multiprocessing
from multiprocessing import Pool

from time import time


def get_feature(data_X_org, org, dest):
    org_data = data_X_org.loc[data_X_org['nearest station']==org]
    org_lat = data_X_org.loc[data_X_org['nearest station']==org, 'INTPTLAT'].values
    org_lon = data_X_org.loc[data_X_org['nearest station']==org, 'INTPTLON'].values

    dest_data = data_X_org.loc[data_X_org['nearest station']==dest]
    dest_lat = data_X_org.loc[data_X_org['nearest station']==dest, 'INTPTLAT'].values
    dest_lon = data_X_org.loc[data_X_org['nearest station']==dest, 'INTPTLON'].values

    dist = np.sqrt((dest_lat-org_lat)**2 + (dest_lon-org_lon)**2) * 111  # km

    dest_data = dest_data.drop(['INTPTLAT', 'INTPTLON', 'nearest station'], axis=1)
    org_data = org_data.drop(['INTPTLAT', 'INTPTLON', 'nearest station'], axis=1)
    dest_data = dest_data.to_numpy()
    org_data = org_data.to_numpy()

    feature = np.concatenate([org_data, dest_data], axis=1)
    feature = np.append(feature, dist)

    return feature


def construct_OD(process_name, data_X_org, from_ind, to_ind, OD_hour, data_X_hour):
    print('Start process' + process_name)
    start_t = time()

    for i in range(len(OD_hour)):
        org = OD_hour.iloc[i, 2]  # org
        dest = OD_hour.iloc[i, 3]  # dest
        feature = get_feature(data_X_org, org, dest)
        data_X_hour[i, :] = feature
        if i % 1e4 == 0:
            print('%s finished  no %i, time spent %.4f'%(process_name, i, time()-start_t))

    print('End process' + process_name)
    return data_X_hour


if __name__ == '__main__':
    OD_hour = pd.read_csv('/Volumes/GoogleDrive/My Drive/Graduate/SP22 CE 299/data/BART/hour data/date-hour-soo-dest-2019.csv', header=None)  # mac
    # OD_hour = pd.read_csv('G:/我的云端硬盘/Graduate/SP22 CE 299/data/BART/hour data/date-hour-soo-dest-2019.csv', header=None)  # windows
    OD_hour.columns = ['date', 'hour', 'org', 'dest', 'trip_count']
    data_X_org = pd.read_csv('./data/data_X.csv', index_col=0).iloc[:48, :]

    # data_X_hour = np.zeros([OD_hour.shape[0], 37])

    n_cpu = multiprocessing.cpu_count()

    interval = len(OD_hour) // n_cpu * np.arange(n_cpu)
    interval = np.append(interval, OD_hour.shape[0])

    pool = Pool(processes=n_cpu)
    params = []
    for i in range(len(interval)-1):
        from_ = interval[i]
        to_ = interval[i+1]
        process_name = 'P' + str(i)
        data_X_hour = np.zeros([to_ - from_, 37])
        OD_hour_ = OD_hour.iloc[from_:to_, :]
        params.append((process_name, data_X_org, from_, to_, OD_hour_, data_X_hour))
        print('Initializing # %i finished'%i)

    bart_OD_set = pool.starmap(func=construct_OD, iterable=params)

    # please set a breakpoint here, then store the data manually
    print('end')