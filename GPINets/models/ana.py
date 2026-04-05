import numpy as np
import torch
from torch import nn


def attention(query, key, value):  #
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5  #
    prob = torch.nn.functional.softmax(scores, dim=-1)  #
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob  #


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):  #
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads  #
        self.num_heads = num_heads

        self.w_qs = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)
        self.w_ks = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)
        self.w_vs = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)
        self.fc = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)

    def forward(self, query, key, value, sc_use=False):
        batch_dim = query.size(0)

        query = self.w_qs(query).view(batch_dim, self.dim, self.num_heads, -1)
        key = self.w_ks(key).view(batch_dim, self.dim, self.num_heads, -1)
        value = self.w_ks(value).view(batch_dim, self.dim, self.num_heads, -1)

        x, prob = attention(query, key, value)
        return self.fc(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1)), prob


def MLP(channels: list, do_bn=True):  #
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _normalization(self, inputs, normal_fn=None):
    c_desc = inputs.permute(0, 2, 1)
    c_desc = normal_fn(c_desc)
    c_desc = c_desc.permute(0, 2, 1)
    return c_desc


class ana_block(nn.Module):
    def __init__(self, channel_num=128):
        super().__init__()

        self.channel_num = channel_num
        self.atten = MultiHeadedAttention(4, self.channel_num)
        conv_channel_base = 2
        self.layernorm = nn.LayerNorm(self.channel_num)

        self.merge = nn.Sequential()
        self.zoom = torch.nn.Parameter(2 * torch.ones((1, 4, 1)), requires_grad=True)
        self.offset = torch.nn.Parameter(torch.ones((1, 4, 1)), requires_grad=True)
        self.neighborconv = MLP([4] + [32, 64, 128] + [128])
        self.n_norm = nn.LayerNorm(128)
        self.conv = MLP([conv_channel_base * self.channel_num, self.channel_num, self.channel_num])

    def forward(self, inputs):
        '''
        input: tensor b(hc)n
        output: tensor b(hc)n
        '''
        c_feat, prob = self.atten(inputs, inputs, inputs)

        H_star = torch.sum(prob, dim=-2, keepdims=True)  # 二阶注意力上下文  一堆公式中选择了一次
        H_star = H_star - (H_star ** 2) / H_star.shape[-1]
        H_star = 1.414 * H_star

        confidence = torch.reciprocal(1 + ((-self.zoom * H_star.squeeze(-2)).exp()))
        n_feat = self.neighborconv(confidence)  # 1/(1+w*exp(-x))  维度从多头数变成128
        n_feat = self.n_norm(n_feat.permute(0, 2, 1)).permute(0, 2, 1)
        inputs = inputs + n_feat

        final_feat = self.conv(torch.cat((inputs, c_feat), dim=1))  # conv就是merge操作(256-128)

        final_feat = self.n_norm(final_feat.permute(0, 2, 1)).permute(0, 2, 1)

        return final_feat


# features = torch.rand(16, 128, 1000)
# model = ana_block()
# out = model(features)
