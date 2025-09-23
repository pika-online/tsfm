# dataset.py (窗口级 CMVN 版本)
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CryptoDataset(Dataset):
    def __init__(self, csv_file, seq_len=60, mode="train", split_ratio=(0.7, 0.15, 0.15)):
        df = pd.read_csv(csv_file)
        data = df[['open','high','low','close','volume','amount']].values.astype(np.float32)

        n = len(data)
        n_train = int(split_ratio[0] * n)
        n_val   = int(split_ratio[1] * n)
        if mode == "train":
            data = data[:n_train]
        elif mode == "val":
            data = data[n_train:n_train+n_val]
        else:
            data = data[n_train+n_val:]

        self.seq_len = seq_len
        self.samples = []
        for i in range(len(data) - seq_len):
            self.samples.append(data[i:i+seq_len])
        self.samples = np.array(self.samples)
        print(self.samples.shape)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]   # shape (T,6)
        ohlc = x[:, :4]
        vol_amt = x[:, 4:]

        # ---- OHLC: 窗口 CMVN ----
        mean_ohlc = np.mean(ohlc, axis=0, keepdims=True)
        std_ohlc  = np.std(ohlc, axis=0, keepdims=True) + 1e-8
        ohlc_norm = (ohlc - mean_ohlc) / std_ohlc

        # ---- Vol/Amount: log1p + 窗口 CMVN ----
        vol_amt = np.log1p(vol_amt)
        mean_va = np.mean(vol_amt, axis=0, keepdims=True)
        std_va  = np.std(vol_amt, axis=0, keepdims=True) + 1e-8
        vol_amt_norm = (vol_amt - mean_va) / std_va

        # 拼接回去
        x_norm = np.concatenate([ohlc_norm, vol_amt_norm], axis=1)

        return torch.tensor(x_norm, dtype=torch.float32), \
               torch.tensor(np.concatenate([mean_ohlc, mean_va], axis=1), dtype=torch.float32), \
               torch.tensor(np.concatenate([std_ohlc, std_va], axis=1), dtype=torch.float32)

def get_dataloader(csv_file, seq_len=60, batch_size=64, mode="train"):
    dataset = CryptoDataset(csv_file, seq_len, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train")), dataset



if __name__ == "__main__":

    dataset = CryptoDataset(csv_file="data/ETHUSDT_1h_20200101_20250901.csv",seq_len=120)
