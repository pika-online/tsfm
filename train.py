# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tsfm.dataset.core import get_dataloader
from tsfm.model.gru import *
import os

def train_model(csv_file, seq_len=60, batch_size=64, epochs=20, lr=1e-3, device="cuda:1"):
    # dataloader + dataset（dataset 里包含scaler）
    train_loader, train_dataset = get_dataloader(csv_file, seq_len, batch_size, mode="train")
    val_loader,   val_dataset   = get_dataloader(csv_file, seq_len, batch_size, mode="val")

    model = GRUVAE(input_dim=6,latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        train_loss = 0
        for x, mean, std in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ---- validate ----
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, mean, std in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=0.1)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---- 保存最好模型 ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "gru_autoencoder.pth")
            print(f"✅ 模型已保存 (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":

    csv_file = "data/ETHUSDT_1h_20200101_20250901.csv"
    train_model(csv_file)
