import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, df:pd.DataFrame, seq_len:int, label_len:int, pred_len:int, scaler:StandardScaler):
        """
        df: [date,[words],OT]
        """

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scaler = scaler

        # 时间特征
        df_stamp = pd.to_datetime(df['date']).to_frame(name='date')
        df_stamp['month'] = df_stamp['date'].dt.month / 12.0
        df_stamp['day'] = df_stamp['date'].dt.day / 31.0
        df_stamp['weekday'] = df_stamp['date'].dt.weekday / 6.0
        df_stamp['hour'] = (df_stamp['date'].dt.hour if hasattr(df_stamp['date'].dt, 'hour') else 0) / 23.0
        self.timestamp = df_stamp.drop(columns=['date']).values.astype(np.float32)

        # 多变量
        self.data = df.iloc[:, 1:].values.astype(np.float32)
        print(f"Before: max: {self.data.max()}, min: {self.data.min()}")
        self.data = self.scaler.transform(self.data) 
        

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.timestamp[s_begin:s_end]
        seq_y_mark = self.timestamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    

if __name__ == "__main__":
    
    df = pd.read_csv('dataset/ETTh1.csv')
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_valid = len(df) - num_train - num_test

    seq_len = 96
    label_len = 48
    pred_len = 24

    # 数据集切片
    border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
    border2s = [num_train, num_train + num_valid, len(df)]
    
    # 数据集划分
    df_train = df[border1s[0]:border2s[0]]
    df_valid = df[border1s[1]:border2s[1]]
    df_test = df[border1s[2]:border2s[2]]
    
    # 训练集scaler
    scaler = StandardScaler()
    scaler.fit(df_train.iloc[:, 1:].values)

    tsd_train = TimeSeriesDataset(df_train, seq_len, label_len, pred_len, scaler)
    tsd_valid = TimeSeriesDataset(df_valid, seq_len, label_len, pred_len, scaler)
    tsd_test = TimeSeriesDataset(df_test, seq_len, label_len, pred_len, scaler)