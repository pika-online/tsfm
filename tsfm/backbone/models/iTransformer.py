from .base import Base_Model
import torch
import torch.nn as nn
from tsfm.backbone.layers.Transformer_EncDec import Encoder, EncoderLayer
from tsfm.backbone.layers.SelfAttention_Family import FullAttention, AttentionLayer


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # 线性层作用在 time 维上：seq_len -> d_model
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x:       B x L x x_dim
        x_mark:  B x L x time_dim  或 None
        输出:    B x token_num x d_model
        """
        x = x.permute(0, 2, 1)  # B x x_dim x L
        # x: [Batch, Variate, Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 拼在变量维：B x (x_dim + time_dim) x L
            x = self.value_embedding(x)
            # x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], dim=1))
        # x: [Batch, Tokens, d_model]
        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """
    多尺度按变量的 depthwise 1D 卷积块，在时间维上提取局部模式。
    输入:  B x L x x_dim
    输出:  B x L x x_dim
    """
    def __init__(self, x_dim, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=x_dim,
                out_channels=x_dim,
                kernel_size=k,
                padding=k // 2,  # 保持长度不变
                groups=x_dim      # depthwise：每个变量自己的卷积核
            )
            for k in kernel_sizes
        ])
        self.act = nn.GELU()

    def forward(self, x):
        # x: B x L x x_dim
        x_t = x.permute(0, 2, 1)  # B x x_dim x L
        outs = []
        for conv in self.convs:
            outs.append(self.act(conv(x_t)))  # B x x_dim x L
        out = sum(outs) / len(outs)          # 多尺度平均融合
        out = out.permute(0, 2, 1)           # B x L x x_dim
        return out



class Model(Base_Model):
    """
    iTransformer + 多尺度时序卷积增强
    通过 agg_type 控制注入方式：
        - "add"   : 残差相加
        - "concat": 拼接 + 线性变换
        - "gate"  : 门控融合
        - "film"  : FiLM 调制
        - "soft"  : 两路 soft 选择
    """

    def __init__(self, **kwargs):
        # 先取出 agg_type，避免传入 Base_Model
        super().__init__(**kwargs)

        agg_type = kwargs.get('agg_type', 'gate')
        x_dim = kwargs['x_dim']
        time_dim = kwargs['time_dim']
        seq_len = kwargs['seq_len']
        label_len = kwargs['label_len']
        pred_len = kwargs['pred_len']
        time_feat = kwargs['time_feat']
        d_model = kwargs['d_model']
        dropout = kwargs['dropout']
        n_heads = kwargs['n_heads']
        d_ff = kwargs['d_ff']
        e_layers = kwargs['e_layers']
        d_layers = kwargs['d_layers']
        attn_dropout = kwargs['attn_dropout']  # 目前没单独用，可按需替换掉下面的 dropout
        activation = kwargs['activation']
        factor = kwargs['factor']
        self.pred_len = pred_len
        self.label_len = label_len

        # -------- 多尺度卷积模块 --------
        self.temporal_block = TemporalConvBlock(x_dim, kernel_sizes=(3, 5, 7))

        # -------- 不同注入方式需要的融合层 --------
        self.agg_type = agg_type
        valid_agg = ['add', 'concat', 'gate', 'film']
        if self.agg_type not in valid_agg:
            raise ValueError(f"agg_type must be one of {valid_agg}, got {self.agg_type}")

        # concat / gate 用：输入维度 2 * x_dim
        self.fusion_linear = nn.Linear(2 * x_dim, x_dim)   # for "concat"
        self.fusion_gate = nn.Linear(2 * x_dim, x_dim)     # for "gate"

        # film 用：由 temporal_feat 生成 gamma / beta
        self.film = nn.Linear(x_dim, 2 * x_dim)            # for "film"

        # soft 用：为每一路 (x, t) 打分
        self.stream_score = nn.Linear(x_dim, 1)            # for "soft"

        # -------- 原 iTransformer 结构 --------
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.dec_embedding = DataEmbedding_inverted(label_len + pred_len, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
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

        self.proj_dec = nn.Linear(d_model, pred_len, bias=True)



    def _aggregate(self, x_enc, temporal_feat):
        """
        根据 agg_type 把 temporal_feat 注入到 x_enc 中。
        x_enc        : B x L x x_dim
        temporal_feat: B x L x x_dim
        返回         : B x L x x_dim
        """
        if self.agg_type == 'add':
            # 残差：简单相加
            return x_enc + temporal_feat

        elif self.agg_type == 'concat':
            # 拼接后线性变换
            fusion_in = torch.cat([x_enc, temporal_feat], dim=-1)  # B x L x 2D
            return self.fusion_linear(fusion_in)                   # B x L x D

        elif self.agg_type == 'gate':
            # 门控融合：g * t + (1-g) * x
            fusion_in = torch.cat([x_enc, temporal_feat], dim=-1)  # B x L x 2D
            gate = torch.sigmoid(self.fusion_gate(fusion_in))      # B x L x D
            return gate * temporal_feat + (1.0 - gate) * x_enc     # B x L x D

        elif self.agg_type == 'film':
            # FiLM 风格调制： (1+γ)*x + β，γ/β 由 temporal_feat 生成
            gamma_beta = self.film(temporal_feat)                  # B x L x 2D
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)       # B x L x D, B x L x D
            return (1.0 + gamma) * x_enc + beta                    # B x L x D

        else:
            # 理论上不会走到这里，上面已经检查过
            raise ValueError(f"Unknown agg_type: {self.agg_type}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Inputs:
            x_enc:      B x seq_len x x_dim          (包括 label_len)
            x_mark_enc: B x seq_len x time_dim       (包括 label_len)
            x_dec:      B x (label_len+pred_len) x x_dim,   x_dec[:, label_len:, :] 为 0
            x_mark_dec: B x (label_len+pred_len) x time_dim
        Outputs:
            dec_out:    B x pred_len x x_dim
        """
        pred_time_mark = x_mark_dec[:, -self.pred_len:, :]

        # 1) 归一化
        x_enc, means, stdev = self._normalize(x_enc)
        B, T, F = x_enc.shape  # F = x_dim

        # 2) 多尺度卷积提取局部时间模式
        temporal_feat = self.temporal_block(x_enc)        # B x L x x_dim

        # 3) 按 agg_type 注入
        x_enc_fused = self._aggregate(x_enc, temporal_feat)  # B x L x x_dim

        # 4) 送入 iTransformer Encoder
        enc_out = self.enc_embedding(x_enc_fused, x_mark_enc)  # B x tokens x d_model
        enc_out, _ = self.encoder(enc_out, attn_mask=None)     # B x tokens x d_model

        # 5) 投影到预测长度，并只保留前 N 个变量 token
        dec_tmp = self.proj_dec(enc_out)                         # B x tokens x pred_len
        dec_out = dec_tmp.permute(0, 2, 1)[:, :, :F]


        # 6) 反归一化
        dec_out = self._denormalize(dec_out, means, stdev)
        return dec_out
