import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, know_num, hid_dim, emb_size, time_type="cat", norm=False, dropout=0.5, adj=None):
        super(DNN, self).__init__()
        self.know_num = know_num
        self.hid_dim = hid_dim
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.adj = adj

        if self.adj is None:
            if self.time_type == "cat":
                in_dim_temp = self.know_num + self.time_emb_dim
            else:
                raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        else:
            pass
            # if self.time_type == "cat":
            #     in_dim_temp = 2 * self.know_num + self.time_emb_dim
            # else:
            #     raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
            # self.learned_adj = nn.Parameter(torch.randn(size=(self.know_num, self.know_num)))
            # self.gcn = GCN(adj=self.learned_adj, nhid=self.hid_dim, dropout=0.1, layers=1)
            # self.k_feature = nn.Parameter(torch.randn(size=(self.know_num, self.hid_dim)))
            # self.W = nn.Sequential(
            #     nn.Linear(self.hid_dim, self.hid_dim),
            #     nn.Tanh(),
            #     nn.Linear(self.hid_dim, self.know_num),
            #     nn.Tanh()
            # )
            # self.activation = nn.LeakyReLU()

        self.layers = nn.Sequential(
            nn.Linear(in_dim_temp, self.hid_dim),
            nn.Tanh(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.Tanh(),
            nn.Linear(self.hid_dim, self.know_num)
        )
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.adj is None:
            if self.norm:
                x = F.normalize(x)
            x = self.drop(x)
            h = torch.cat([x, emb], dim=-1)
            h = self.layers(h)
        else:
            pass
            # if self.norm:
            #     x = F.normalize(x)
            # x = self.drop(x)
            # gcn_agg = self.activation(self.gcn(self.k_feature)).mean(dim=0)
            # temp_x = x * self.W(gcn_agg) + x
            # if self.norm:
            #     temp_x = F.normalize(temp_x)
            # temp_x = self.drop(temp_x)
            # h = torch.cat([temp_x, x, emb], dim=-1)
            # h = self.layers(h)
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, adj, nhid, dropout=0.1, layers=5):
        super(GCN, self).__init__()
        self.layers = layers
        self.gcs = nn.ModuleList([GraphConvolution(nhid, nhid) for _ in range(layers)])
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.adj = adj

    def forward(self, x):
        if self.layers > 1:
            # x : batch_size * nhid
            res = []
            for i, gc in enumerate(self.gcs):
                res.append(self.dropout(self.activation(gc(x, self.adj))))
            return torch.stack(res, dim=1).mean(dim=1)
        else:
            res = []
            for i, gc in enumerate(self.gcs):
                res.append(self.dropout(self.activation(gc(x, self.adj))))
            return res[0]
