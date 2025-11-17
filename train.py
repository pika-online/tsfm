import random
from tsfm.data_loader import TimeSeriesDataset
import torch
import pandas as pd
import numpy as np
import importlib
import os 
from sklearn.preprocessing import StandardScaler

# 固定随机种子
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        

def load_data(file_path, seq_len, label_len, pred_len, split_ratio=[0.7, 0.1, 0.2]):
    df = pd.read_csv(file_path)
    num_train = int(len(df) * split_ratio[0])
    num_test = int(len(df) * split_ratio[2])
    num_vali = len(df) - num_train - num_test

    # 数据集切片
    border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df)]

    df_train = df[border1s[0]:border2s[0]]
    df_valid = df[border1s[1]:border2s[1]]
    df_test = df[border1s[2]:border2s[2]]
    print(f"Train shape: {df_train.shape}, Valid shape: {df_valid.shape}, Test shape: {df_test.shape}")
    # 训练集scaler
    scaler = StandardScaler()
    scaler.fit(df_train.iloc[:, 1:].values)
    # print(scaler.mean_, scaler.scale_)
    

    tsd_train = TimeSeriesDataset(df_train, seq_len, label_len, pred_len, scaler)
    tsd_valid = TimeSeriesDataset(df_valid, seq_len, label_len, pred_len, scaler)
    tsd_test = TimeSeriesDataset(df_test, seq_len, label_len, pred_len, scaler)
    
    return tsd_train, tsd_valid, tsd_test, scaler


def train_one_epoch(model, train_loader, pred_len, label_len, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(device)
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

        # forward
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = 0
        outputs = outputs[:, -pred_len:, f_dim:]
        batch_y = batch_y[:, -pred_len:, f_dim:]

        # loss 
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def eval_one_epoch(model, valid_loader, pred_len, label_len, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

            # forward
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:]

            # loss 
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
    
    avg_loss = total_loss / len(valid_loader)
    preds = np.concatenate(preds,axis=0)
    trues = np.concatenate(trues,axis=0)
    # print(f"eval_one_epoch: preds shape: {preds.shape}, trues shape: {trues.shape}")
    
    return avg_loss, preds, trues


def metric(pred, true):
    
    def MAE(pred, true):
        return np.mean(np.abs(true - pred))

    def MSE(pred, true):
        return np.mean((true - pred) ** 2)

    mae = MAE(pred, true)
    mse = MSE(pred, true)


    return mae, mse


def exp(
    csv_file:str,
    model_name:str,
    data_name:str,
    configs:dict,
    exp_dir:str
):
    
    # 数据加载
    seq_len = configs["dataset"]['seq_len']
    label_len = configs["dataset"]['label_len']
    pred_len = configs["dataset"]['pred_len']
    split_ratio = configs["dataset"]['split_ratio']
    batch_size = configs["training"]['batch_size']
    tsd_train, tsd_valid, tsd_test, scaler = load_data(csv_file, seq_len, label_len, pred_len, split_ratio)
    train_loader = torch.utils.data.DataLoader(tsd_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(tsd_valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(tsd_test, batch_size=batch_size, shuffle=False)
    
    
    # 模型初始化
    use_time = configs["model"]['use_time']
    input_feat_dim = tsd_train.data.shape[-1]
    time_feat_dim = tsd_train.timestamp.shape[-1]
    model_config = {
        'c_in': input_feat_dim,
        'c_time': time_feat_dim,
        'c_out': input_feat_dim,
        'pred_len': pred_len,
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'activation': 'gelu',
        'factor': 3,
        'use_time': use_time
    }
    print(model_config)
        
    model_module = importlib.import_module(f'tsfm.backbone.models.{model_name}')
    Model = model_module.Model
    model_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'model_{data_name}.pth')
    
    # 训练
    device = configs["training"]['device']
    learning_rate = configs["training"]['learning_rate']
    patience = configs["training"]['patience']
    train_epochs = configs["training"]['train_epochs']
    model = Model(**model_config).to(device)
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    

    for epoch in range(train_epochs):
        loss_train = train_one_epoch(model, train_loader, pred_len, label_len, criterion, model_optim, device, epoch)
        print(f'=====> Epoch {epoch+1}, Train Loss: {loss_train:.6f}')
        
        loss_valid, _, _ = eval_one_epoch(model, valid_loader, pred_len, label_len, criterion, device)
        print(f'=====> Epoch {epoch+1}, Valid Loss: {loss_valid:.6f}')
        
        # loss_test, _, _ = eval_one_epoch(model, test_loader, pred_len, label_len, criterion, device)
        # print(f'=====> Epoch {epoch+1}, Test Loss: {loss_test:.6f}')
        
        early_stopping(loss_valid, model, model_path)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        adjust_learning_rate(model_optim, epoch+1, learning_rate)
        
    # 测试
    model.load_state_dict(torch.load(model_path))
    test_loss, test_preds, test_trues = eval_one_epoch(model, test_loader, pred_len, label_len, criterion, device)
    # shape = test_preds.shape
    # test_preds = scaler.inverse_transform(test_preds.reshape(shape[0]*shape[1],-1)).reshape(shape)
    # test_trues = scaler.inverse_transform(test_trues.reshape(shape[0]*shape[1],-1)).reshape(shape)
    mae, mse = metric(test_preds, test_trues)
    print(f'Test LOSS:{test_loss:.4f} Test MAE: {mae:.4f}, MSE: {mse:.4f}')
    
    return mae,mse
    
    
    
if __name__ == "__main__":

    configs = {
        "dataset":{
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 96,
            'split_ratio': [0.7, 0.1, 0.2]
        },
        "training":{
            'device': "cuda",
            'learning_rate': 0.0001,
            'patience': 3,
            'train_epochs': 5,
            'batch_size': 32,
        },
        "model":{
            'use_time': False,
        }
    }

    model_name = 'Transformer'
    # model_name = 'Informer'
    use_time = configs['model']['use_time']
    exp_dir = f"exp/{model_name}@use_time={use_time}"
    os.makedirs(exp_dir, exist_ok=True)
    result_path = f"{exp_dir}/results.csv"
    
    
    data_dir = 'dataset/'
    results = []
    for data_name in os.listdir(data_dir):
        if data_name.endswith('.csv'):
            csv_file = os.path.join(data_dir, data_name)
            mae, mse = exp(csv_file, model_name, data_name, configs, exp_dir)
            print(f'Dataset: {data_name}, Model: {model_name}, MAE: {mae:.4f}, MSE: {mse:.4f}')
            print('============================================================\n')
            results.append({
                'dataset': data_name,
                'model': model_name,
                'mae': f"{mae:.3f}",
                'mse': f"{mse:.3f}"
            })
    
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(result_path, index=False)
        print(f'All experiment results saved to: {result_path}')