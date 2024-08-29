import pandas as pd
import numpy as np
import glob
import random
import torch
import os
from sklearn.preprocessing import MinMaxScaler
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed: int = 1234):
    """set a fix random seed.
    
    Args:
        seed (int, optional): random seed. Defaults to 9.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
   
def load_data(path, postfix, choose=None):
    files = sorted(glob.glob(path + postfix))
    # print(files)
    if type(choose) is int:
        # random_choose = np.random.randint(0, len(files))
        print(f"building: {files[choose]}")
        df = pd.read_csv(files[choose])
        return df
    elif type(choose) is str:
        file = glob.glob(path + choose + postfix)[0]
        print(f"building: {file}")
        df = pd.read_csv(file)
        return df
    elif type(choose) is list:
        dfs = []
        for file in choose:
            if type(file) is int:
                print(f"building: {files[file]}")
                dfs.append(pd.read_csv(files[file]))
            elif type(file) is str:
                file = glob.glob(path + file + postfix)[0]
                dfs.append(pd.read_csv(file))
        return dfs
    elif choose is None:
        random_choose = np.random.randint(0, len(files))
        # print(f"building: {files[random_choose]}")
        df = pd.read_csv(files[random_choose])
        return df
    else:
        dfs = []
        for file in files:
            dfs.append(pd.read_csv(file))
        return dfs, files

def reshape_to_3d(df, hours_per_day=24):
    data_3d = []
    num_days = len(df) // hours_per_day
    for i in range(num_days):
        daily_data = df.iloc[i*hours_per_day:(i+1)*hours_per_day, 1:].values
        data_3d.append(daily_data)
    return np.array(data_3d)

def construct_dataset(data):
    
    # divide the dataset
    train_data = data[(data['hour'] >= '2019-06-01') & (data['hour'] < '2019-08-01')]
    test_data = data[(data['hour'] >= '2019-08-01') & (data['hour'] < '2019-09-01')]
    
    train_data_normalized = train_data.copy()
    test_data_normalized = test_data.copy()
    train_data_normalized['t'] = train_data['t'] / 20
    train_data_normalized['price'] = train_data['price'] * 20
    train_data_normalized['temperature'] = train_data['temperature'] / 20
    test_data_normalized['t'] = test_data['t'] / 20
    test_data_normalized['price'] = test_data['price'] * 20
    test_data_normalized['temperature'] = test_data['temperature'] / 20
    
    train_data = reshape_to_3d(train_data_normalized)
    test_data = reshape_to_3d(test_data_normalized)

    return train_data, test_data
