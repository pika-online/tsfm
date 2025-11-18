from .base import Base_Model
import torch.nn as nn
from tsfm.backbone.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from tsfm.backbone.layers.SelfAttention_Family import FullAttention, AttentionLayer
from tsfm.backbone.layers.Embedding import DataEmbedding, DataEmbedding_inverted



class Model(Base_Model):
    """
    iTransformer 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_in = kwargs['c_in']
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
        seq_len = kwargs['seq_len']

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        x_enc, means, stdev = self._normalize(x_enc)
        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        dec_out = self._denormalize(dec_out, means, stdev)
        return dec_out[:, -self.pred_len:, :]