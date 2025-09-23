import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime

def cmvn(x: np.ndarray):
    eps = 1e-8
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)

def date2ts(date_str: str):
    """
    输入: '20250101'
    输出: 毫秒级时间戳
    """
    dt = datetime.strptime(date_str, "%Y%m%d")
    return int(dt.timestamp() * 1000)

def read_csv(csv_file,):
    """
    数据示例：
    timestamps,open,high,low,close,volume,amount
    1577836800000,0.03285,0.03285,0.0327,0.03273,875666.2,28677.431516
    1577837100000,0.03276,0.03279,0.03275,0.03279,52921.6,1734.292968
    1577837400000,0.03279,0.03279,0.03277,0.03278,16146.5,529.285568
    """
    df = pd.read_csv(csv_file)
    timestamps = df["timestamps"].to_list()
    data = df[['open','high','low','close','volume','amount']].values.astype(np.float32)
    print(f"成功读取:{csv_file}")
    return timestamps, data

def create_windows(timestamps, data, win_size, win_shift, split_date=None):
    stop_ts = date2ts(split_date) if split_date is not None else np.inf
    windows1 = []
    windows2 = []
    for i in range(win_size, len(data), win_shift):
        win = data[i - win_size:i]
        date = timestamps[i - 1]  # UTC 毫秒
        ohlc = cmvn(win[:, :4])              # open, high, low, close
        va = cmvn(np.log1p(win[:, 4:]))      # volume, amount (对数变换后标准化)
        win_norm = np.concatenate([ohlc, va], axis=1)
        if date <= stop_ts:
            windows1.append(win_norm)
        else:
            windows2.append(win_norm)
    return np.array(windows1, dtype=np.float32), np.array(windows2, dtype=np.float32)


def create_windows_for_multifiles(csv_files:list,win_size, win_shift, split_date=None):
    wins1, wins2 = [], []
    for csv_file in csv_files:
        timestamps, data = read_csv(csv_file)
        win1, win2 = create_windows(timestamps, data, win_size=win_size, win_shift=win_shift, split_date=split_date)
        print(f"win1:{win1.shape}, win2:{win2.shape}")
        wins1.append(win1)
        wins2.append(win2)
    return np.concatenate(wins1,axis=0), np.concatenate(wins2,axis=0)


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        data: (N, win_size, feature_dim) numpy 数组
        """
        self.data = torch.from_numpy(data)  # 转成 tensor，避免反复转换
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 这里不需要 label，因为 VAE 是自编码
        return self.data[idx]

if __name__ == "__main__":

    
    import os 
    dir = "data"
    csv_files = [f"{dir}/{filename}" for filename in os.listdir(dir)]
    
    wins1, wins2 = create_windows_for_multifiles(csv_files, win_size=120, win_shift=2, split_date="20250601")
    np.savez("samples/dataset.npz", TR=wins1, CV=wins2)
