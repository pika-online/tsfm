
import torch
import torch.nn as nn
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
    """
    d_feat: 输入数值特征维度
    d_time: 时间特征维度（比如 hour, weekday 等 one-hot/embedding 后的维度）
    d_model: Transformer hidden size
    mode:
        - "none"   : 只用数值 + 位置
        - "add"    : 直接加和时间 embedding
        - "concat" : 拼接后再过 Conv
        - "gate"   : 时间特征做门控
        - "film"   : FiLM 风格，用时间特征做 scale & shift
        - "add_gate": 在 add 的基础上再加一个全局可学习门
    """
    def __init__(self, d_feat, d_time, d_model, dropout=0.1, mode="add"):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=d_feat, d_model=d_model)
        self.value_embedding2 = TokenEmbedding(c_in=d_feat + d_time, d_model=d_model)

        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 时间特征 embedding + 相关线性层
        self.temporal_embedding = TimeFeatureEmbedding(d_inp=d_time, d_model=d_model)
        self.gate_linear = nn.Linear(d_time, d_model)

        # FiLM: 用时间特征生成 gamma / beta
        self.film_gamma = nn.Linear(d_time, d_model)
        self.film_beta = nn.Linear(d_time, d_model)

        # 全局门（标量），控制时间特征整体强度
        self.global_gate = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode

    def forward(self, x, x_mark):
        """
        x:      [B, L, d_feat]
        x_mark: [B, L, d_time] or None
        """
        # 基础 embedding：数值 + 位置
        base = self.value_embedding(x) + self.position_embedding(x)

        # 如果没有时间特征，直接返回基础 embedding
        if x_mark is None or self.mode == "none":
            return self.dropout(base)

        if self.mode == "add":
            # 直接相加
            t = self.temporal_embedding(x_mark)      # [B, L, d_model]
            out = base + t

        elif self.mode == "concat":
            # 数值和时间拼接，然后再做 Conv
            x_cat = torch.cat([x, x_mark], dim=-1)   # [B, L, d_feat + d_time]
            out = self.value_embedding2(x_cat) + self.position_embedding(x)

        elif self.mode == "gate":
            # 时间特征做门控
            t = self.temporal_embedding(x_mark)      # [B, L, d_model]
            g = torch.sigmoid(self.gate_linear(x_mark))  # [B, L, d_model]
            out = base + g * t

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.dropout(out)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)