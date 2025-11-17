import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, df:pd.DataFrame, seq_len:int, label_len:int, pred_len:int):
        """
        df: [date,[words],OT]
        """

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        df_stamp = pd.to_datetime(df['date'])
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)/13.0
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)/32.0
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)/7.0
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)/24.0
        self.df_stamp = df_stamp.drop(['date'], axis=1).values

        self.data = df.columns[1:].values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.df_stamp[s_begin:s_end]
        seq_y_mark = self.df_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
if __name__ == "__main__":
    
    df = pd.read_csv('dataset/ETTh1.csv')
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test

    seq_len = 96
    label_len = 48
    pred_len = 24

    # 数据集切片
    border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df)]
    
    df_train = df[border1s[0]:border2s[0]]
    df_valid = df[border1s[1]:border2s[1]]
    df_test = df[border1s[2]:border2s[2]]

    tsd_train = TimeSeriesDataset(df_train, seq_len, label_len, pred_len)
    tsd_valid = TimeSeriesDataset(df_valid, seq_len, label_len, pred_len)
    tsd_test = TimeSeriesDataset(df_test, seq_len, label_len, pred_len)