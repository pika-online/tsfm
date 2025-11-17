import torch
import torch.nn as nn
from tsfm.backbone.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from tsfm.backbone.layers.SelfAttention_Family import  AttentionLayer, ProbAttention
from .base import Base_Model
from tsfm.backbone.layers.Embedding import DataEmbedding



class Model(Base_Model):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, **kwargs
    ):
        super().__init__(**kwargs)
        self.pred_len = kwargs['pred_len']
        self.use_time = kwargs['use_time']
        c_in = kwargs['c_in']
        c_time = kwargs['c_time']
        d_model = kwargs['d_model']
        dropout = kwargs['dropout']
        n_heads = kwargs['n_heads']
        d_ff = kwargs['d_ff']
        e_layers = kwargs['e_layers']
        d_layers = kwargs['d_layers']
        factor = kwargs['factor']
        activation = kwargs['activation']
        c_out = kwargs['c_out']
        
        
        # Embeddings
        self.enc_embedding = DataEmbedding(c_in, c_time, d_model, dropout=dropout)
        self.dec_embedding = DataEmbedding(c_in, c_time, d_model, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )


    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        x_enc, means, stdev = self._normalize(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = self._denormalize(dec_out, means, stdev)
        return dec_out