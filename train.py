from tsfm.data_loader import TimeSeriesDataset
import torch
import pandas as pd
import time 

from tsfm.backbone.models.Transformer import Model

if __name__ == "__main__":

    # 数据集加载
    df = pd.read_csv('dataset/ETTh1.csv')
    split_ratio = [0.7, 0.1, 0.2]
    num_train = int(len(df) * split_ratio[0])
    num_test = int(len(df) * split_ratio[2])
    num_vali = len(df) - num_train - num_test

    seq_len = 96
    label_len = 48
    pred_len = 96

    # 数据集切片
    border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df)]

    df_train = df[border1s[0]:border2s[0]]
    df_valid = df[border1s[1]:border2s[1]]
    df_test = df[border1s[2]:border2s[2]]

    device = "cuda"
    learning_rate = 0.0001
    train_epochs = 3
    batch_size = 32
    input_feat_dim = len(df.columns) - 1  # 减去时间列
    time_feat_dim = 4  # month, day, weekday, hour
    d_model = 512
    model_config = {
        'c_in': input_feat_dim,
        'c_time': time_feat_dim,
        'c_out': input_feat_dim,
        'pred_len': pred_len,
        'd_model': d_model,
        'n_heads': 8,
        'd_ff': 2048,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'activation': 'gelu',
        'factor': 3
    }

    tsd_train = TimeSeriesDataset(df_train, seq_len, label_len, pred_len)
    tsd_valid = TimeSeriesDataset(df_valid, seq_len, label_len, pred_len)
    tsd_test = TimeSeriesDataset(df_test, seq_len, label_len, pred_len)

    train_loader = torch.utils.data.DataLoader(tsd_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(tsd_valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(tsd_test, batch_size=batch_size, shuffle=False)

    # 模型初始化
    model = Model(**model_config)
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
