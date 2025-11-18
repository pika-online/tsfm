from .base import Base_Model
import torch.nn as nn
from tsfm.backbone.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from tsfm.backbone.layers.SelfAttention_Family import FullAttention, AttentionLayer
from tsfm.backbone.layers.Embedding import DataEmbedding
from tsfm.backbone.layers.Conv_Blocks import Inception_Block_V1
import torch
import torch.nn.functional as F

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res



class Model(Base_Model):
    """
    TimesNet
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_in = kwargs['c_in']
        seq_len = kwargs['seq_len']
        pred_len = kwargs['pred_len']
        top_k = kwargs['top_k']
        num_kernels = kwargs['num_kernels']
        self.pred_len = kwargs['pred_len']
        self.use_time = kwargs['use_time']
        c_time = kwargs['c_time']
        d_model = kwargs['d_model']
        dropout = kwargs['dropout']
        n_heads = kwargs['n_heads']
        d_ff = kwargs['d_ff']
        e_layers = kwargs['e_layers']
        d_layers = kwargs['d_layers']
        attn_dropout = kwargs['attn_dropout']
        activation = kwargs['activation']
        factor = kwargs['factor']
        c_out = kwargs['c_out']

        self.e_layers = e_layers
        self.layer_norm = nn.LayerNorm(d_model)

        # Embeddings
        self.enc_embedding = DataEmbedding(c_in, c_time, d_model, dropout=dropout)
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels) for _ in range(e_layers)])
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        x_enc, means, stdev = self._normalize(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        dec_out = self._denormalize(dec_out, means, stdev)
        return dec_out[:, -self.pred_len:, :]