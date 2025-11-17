import torch
import torch.nn as nn
import torch.nn.functional as F
from tsfm.backbone.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from tsfm.backbone.layers.SelfAttention_Family import FullAttention, AttentionLayer
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_inp, d_model):
        super().__init__()
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, d_feat, d_time, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=d_feat, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_inp=d_time, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, use_time=True):
        if use_time is None:
            out = self.value_embedding(x) + self.position_embedding(x)
        else:
            out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(out)


class Model(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(
        self,
        c_in,
        c_time,
        c_out,
        pred_len,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        e_layers=2,
        d_layers=1,
        dropout=0.1,
        attn_dropout=0.1,
        activation='gelu',
        factor=1,
        use_time=True
    ):
        super().__init__()
        self.pred_len = pred_len
        self.use_time = use_time

        # Embeddings
        self.enc_embedding = DataEmbedding(c_in, c_time, d_model, dropout=dropout)
        self.dec_embedding = DataEmbedding(c_in, c_time, d_model, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=attn_dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=attn_dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=attn_dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )


    def _normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev
    
    def _denormalize(self, x, means, stdev):
        x = x * stdev + means
        return x
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        x_enc, means, stdev = self._normalize(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc, use_time=self.use_time)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec, use_time=self.use_time)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = self._denormalize(dec_out, means, stdev)
        return dec_out