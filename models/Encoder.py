import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class MLP_Encoder(nn.Module):
    """
        A MLP based encoder to encode user interactions
    """

    def __init__(self, user_num, exer_num, know_num, embed_dim=512):
        super(MLP_Encoder, self).__init__()
        self.user_num = user_num
        self.exer_num = exer_num
        self.know_num = know_num
        self.embed_dim = embed_dim

        self.user_enc = nn.Sequential(
            nn.Linear(exer_num, embed_dim),
            nn.Sigmoid(),
            nn.Linear(embed_dim, know_num),
            nn.Sigmoid()
        )

        self.time_enc = nn.Sequential(
            nn.Linear(exer_num, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.ReLU()
        )

        self.item_enc = nn.Sequential(
            nn.Linear(user_num, embed_dim),
            nn.Sigmoid(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
            nn.Linear(embed_dim, know_num),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        user_emb = self.user_enc(user)
        item_emb = self.item_enc(item)
        time = self.time_enc(user)
        return user_emb, item_emb, time.view(-1)


class Trans_Encoder(nn.Module):
    """
        A transformer based encoder to encode user interactions.
    """

    def __init__(self, exer_num, knowledge_num, embed_dim):
        super(Trans_Encoder, self).__init__()
        self.seq_len = exer_num
        self.output_dim = knowledge_num
        self.embedding = nn.Embedding(exer_num, knowledge_num)
        self.trans = nn.Linear(knowledge_num, embed_dim)
        self.net_0 = MultiHeadAttention(embed_dim=embed_dim, num_heads=1)
        self.net_1 = MultiHeadAttention(embed_dim=embed_dim, num_heads=1)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * embed_dim, knowledge_num),
            nn.Tanh()
        )

        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, data_0, know_0, data_0_length, data_1, know_1, data_1_length):
        emb_0 = self.trans(self.embedding(data_0) * know_0)
        emb_1 = self.trans(self.embedding(data_1) * know_1)
        data_0_mask = self.data_mask(data_0, data_0_length + 1)
        data_1_mask = self.data_mask(data_1, data_1_length + 1)

        data_0_sa = self.net_0(query=emb_0, key=emb_0, value=emb_0, mask=data_0_mask)
        data_1_sa = self.net_1(query=emb_1, key=emb_1, value=emb_1, mask=data_1_mask)

        data_0_sa = data_0_sa.sum(dim=1)
        data_1_sa = data_1_sa.sum(dim=1)

        input_emb = torch.cat([data_0_sa, data_1_sa], dim=-1)
        output_emb = self.output_layer(input_emb)
        return output_emb

    def reparameterization(self):
        # if Encode to Normal Distribution
        pass

    def data_mask(self, src, s_len):
        if type(src) == torch.Tensor:
            mask = torch.zeros_like(src)
            for i in range(len(src)):
                mask[i, :s_len[i]] = 1
        return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.scaling = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, exer_num, embed_dim)
        # key: (batch_size, exer_num, embed_dim)
        # value: (batch_size, exer_num, embed_dim)
        # mask: (batch_size, exer_num, seq_len_k)
        batch_size = query.size(0)
        # project query, key and value to multi-heads
        # shape: (batch_size, seq_len_*, num_heads, head_dim)
        query = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # compute scaled dot-product attention
        # shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(query * self.scaling, key.transpose(-2, -1))

        if mask is not None:
            # mask: (batch_size, 1 ,seq_len_q ,seq_len_k)
            bs = mask.shape[0]
            max_seq_len = mask.shape[1]
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1)
            mask = mask.expand(bs, 1, max_seq_len, max_seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float(-1e9))

        # shape: (batch_size ,num_heads ,seq_len_q ,seq_len_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # apply dropout
        attn_weights = self.dropout(attn_weights)

        # compute weighted sum of values
        # shape: (batch_size ,num_heads ,seq_len_q ,head_dim)
        attn_output = torch.matmul(attn_weights, value)

        # concatenate heads and project output
        # shape: (batch_size ,seq_len_q ,embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
