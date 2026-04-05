import torch
from torch import nn


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # inner[32,2000,2000]内积？
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # xx[32,1,2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # distance[32,2000,2000]****记得回头看

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k) [32,2000,9] [32,1000,6]

    return idx[:, :, :]


def get_graph_feature(x, k=9, idx=None):
    # x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # x[32,128,2000]
    if idx is None:
        idx_out = knn(x, k=k)  # idx_out[32,2000,9]
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base  # idx[32,2000,9] 把32个批次的标号连续了

    idx = idx.view(-1)  # idx[32*2000*9] 把32个批次连在一起了 [32*1000*6]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # x[32,2000,128]
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)  # feature[32,2000,9,128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # x[32,2000,9,128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()  # feature[32,256,2000,9] 图特征
    return feature


class GCN(nn.Module):
    def __init__(self, in_channel):
        super(GCN, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def gcn(self, x, w):
        B, _, N, _ = x.size()

        with torch.no_grad():
            w = torch.relu(torch.tanh(w)).unsqueeze(-1)
            A = torch.bmm(w, w.transpose(1, 2))
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
        out = self.gcn(x, w)
        out = self.conv(out)
        return out


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


def attention(query, key, value):  #
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5  #
    prob = torch.nn.functional.softmax(scores, dim=-1)  #
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob  #


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


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


class GraphAttentionNet(nn.Module):
    def __init__(self, out_channel):
        super(GraphAttentionNet, self).__init__()
        self.in_channel = out_channel
        self.out_channel = out_channel
        self.res1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.res2 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.down1 = diff_pool(self.out_channel, 250)
        self.l1 = []
        for _ in range(2):
            self.l1.append(OAFilter(self.out_channel, 250))
        self.up1 = diff_unpool(self.out_channel, 250)
        self.l1 = nn.Sequential(*self.l1)

        self.embedding = GCN(128)
        self.mlp1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
            # [32,128,2000,9]→[32,128,2000,3]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),  # [32,128,2000,3]→[32,128,2000,1]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

        self.atten = MultiHeadedAttention(4, 128)

        self.zoom = torch.nn.Parameter(2 * torch.ones((1, 4, 1)), requires_grad=True)
        self.offset = torch.nn.Parameter(torch.ones((1, 4, 1)), requires_grad=True)
        self.neighborconv = MLP([4] + [32, 64, 128] + [128])
        self.n_norm = nn.LayerNorm(128)
        self.layernorm = nn.LayerNorm(128)

        self.conv = MLP([256, 128, 128])

    def _sc_process(self, prob):
        # if self.config['hstars_type'] == 1:

        H_star = torch.sum(prob, dim=-2, keepdims=True)
        H_star = H_star - (H_star ** 2) / H_star.shape[-1]
        H_star = 1.414 * H_star

        # elif self.config['hstars_type'] == 3:  # sum-sum(^2)
        #     H_star = torch.sum(prob, dim=-2, keepdims=True) - torch.sum(prob ** 2, dim=-2, keepdims=True)
        return H_star

    def _normalization(self, inputs, normal_fn=None):
        '''
        input: tensor BCN
        output: tensor BCN
        '''

        if normal_fn is None:
            c_desc = inputs.permute(0, 2, 1)
            c_desc = self.layernorm(c_desc)
            c_desc = c_desc.permute(0, 2, 1)
            return c_desc
        else:
            c_desc = inputs.permute(0, 2, 1)
            c_desc = normal_fn(c_desc)
            c_desc = c_desc.permute(0, 2, 1)
            return c_desc

    def forward(self, x):
        B, _, N, _ = x.shape

        # out = x.transpose(1, 3).contiguous()
        out = self.res1(x)

        x_down = self.down1(out)
        x2 = self.l1(x_down)
        out = self.up1(out, x2)

        w = self.mlp1(out).view(B, -1)
        out = self.embedding(out, w)

        out = self.res2(out).squeeze(-1)
        # out = get_graph_feature(out)
        #
        # out = self.conv1(out).squeeze(-1)
        # # out = x.squeeze(-1)
        # c_feat, prob = self.atten(out, out, out)
        #
        # H_star = self._sc_process(prob)  # 二阶注意力上下文  一堆公式中选择了一次
        #
        # # 非线性映射
        # confidence = torch.reciprocal(1 + ((-self.zoom * H_star.squeeze(-2)).exp()))
        # n_feat = self.neighborconv(confidence)  # 1/(1+w*exp(-x))  维度从多头数变成128
        # n_feat = self._normalization(n_feat, self.n_norm)
        # inputs = out + n_feat
        #
        # final_feat = self.conv(torch.cat((inputs, c_feat), dim=1))  # conv就是merge操作(256-128)
        #
        # out = self._normalization(final_feat)

        return out


input = torch.randn(16, 128, 1000, 1)
model = GraphAttentionNet(128)
out = model(input)
