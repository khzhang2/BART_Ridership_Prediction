import numpy as np
import pandas as pd

import multiprocessing
from multiprocessing import Pool
from datetime import date


def construct_OD(process_name, from_ind, to_ind, bart_data, stop_table, bart_OD):
    print('Start process' + process_name)
    
    for i in bart_data.index[from_ind:to_ind]:
        # [from_ind, to_ind)
        
        d = bart_data.loc[i, 'Date']
        doy = get_doy(d)  # day of the year, starting from 1
        doy -= 1
        hod = bart_data.loc[i, 'Hour']  # hour of the day, starting from 0
        hour_abs = doy * 24 + hod  # hour index, the 3rd dim of bart_OD

        org = bart_data.loc[i, 'Origin']
        org_ind = stop_table.loc[org, 'stop_index']
        dest = bart_data.loc[i, 'Dest']
        dest_ind = stop_table.loc[dest, 'stop_index']

        pax_flow = bart_data.loc[i, 'Count']

        bart_OD[org_ind, dest_ind, hour_abs] = pax_flow
        
    print('End process' + process_name)
    return bart_OD


def get_doy(d):
    # input date, get day of the year
    return date.fromisoformat(d).timetuple().tm_yday


if __name__ == '__main__':
    bart_path = '/Volumes/Google Drive/My Drive/Graduate/SP22 CE 299/data/BART/hour data/date-hour-soo-dest-2020.csv'
    bart_data = pd.read_csv(bart_path, header=None)
    bart_data.columns = ['Date', 'Hour', 'Origin', 'Dest', 'Count']
    bart_data.head(2)

    stops = bart_data['Origin'].drop_duplicates()
    num_stops = stops.shape[0]  # =50
    # dims: (org, dest, time[hr])
    bart_OD = np.zeros([num_stops, num_stops, 24*366])  # 366 or 365 days

    stop_table = pd.DataFrame(
        range(num_stops),
        index=stops.values,
        columns=['stop_index']
    )

    num_interval = multiprocessing.cpu_count()
    interval = len(bart_data)//num_interval * np.arange(num_interval)
    interval = np.append(interval, bart_data.shape[0])
    interval

    n_cpu = num_interval

    pool = Pool(processes=n_cpu)
    params = []
    for i in range(len(interval)-1):
        from_ = interval[i]
        to_ = interval[i+1]
        process_name = 'P' + str(i)
        params.append((process_name, from_, to_, bart_data, stop_table, bart_OD))

    bart_OD_set = pool.starmap(func=construct_OD, iterable=params)

    # please set a breakpoint here, then store the data manually
    print('end')