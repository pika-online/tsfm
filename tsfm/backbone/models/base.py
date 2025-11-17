import torch.nn as nn
import torch


class Base_Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def _normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev
    
    def _denormalize(self, x, means, stdev):
        x = x * stdev + means
        return x