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
