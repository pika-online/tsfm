# model.py (VAE 版 GRU 自编码器)
import torch
import torch.nn as nn

class GRUVAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32, num_layers=1):
        super(GRUVAE, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # 输出均值和 logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, input_dim, num_layers, batch_first=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        _, h = self.encoder(x)          # (num_layers, B, hidden_dim)
        h = h[-1]                       # 取最后一层
        mu = self.fc_mu(h)              # (B, latent_dim)
        logvar = self.fc_logvar(h)      # (B, latent_dim)
        z = self.reparameterize(mu, logvar)

        h_dec = self.decoder_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h_dec)
        return out, mu, logvar

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    # 重建误差
    recon_loss = nn.MSELoss()(x_hat, x)
    # KL 散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss