import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class MLP_Decoder(nn.Module):
    def __init__(self, know_num, embed_dim):
        super(MLP_Decoder, self).__init__()
        self.know_num = know_num
        self.embed_dim = embed_dim

        self.user_proj = nn.Linear(know_num, know_num)
        self.item_proj = nn.Linear(know_num, know_num)

        self.net = nn.Sequential(
            nn.Linear(know_num, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_emb, item_emb, know_concept):
        know_user_emb = self.user_proj(user_emb * know_concept)
        know_item_emb = self.item_proj(item_emb * know_concept)
        input = know_user_emb - know_item_emb
        return self.net(input)

class IRT_Decoder(nn.Module):
    def __init__(self, know_num):
        super(IRT_Decoder, self).__init__()
        self.theta = nn.Linear(know_num, 1)
        self.b = nn.Linear(know_num, 1)
        self.a = nn.Linear(know_num, 1)

    def forward(self, user_emb, item_emb, know_concept):
        theta = self.theta(user_emb)
        b = self.b(item_emb)
        a = self.a(item_emb)
        return torch.sigmoid(a* (theta - b))

class MIRT_Decoder(nn.Module):
    def __init__(self, know_num, latent_dim):
        super(MIRT_Decoder, self).__init__()
        self.thetas = nn.Linear(know_num, latent_dim)
        self.alphas = nn.Linear(know_num, latent_dim)
        self.betas = nn.Linear(know_num, 1)

    def forward(self, user_emb, item_emb, know_concepts):
        theta = self.thetas(user_emb)
        alpha = self.alphas(item_emb)
        betas = self.betas(item_emb)
        pred = torch.sigmoid((alpha * theta).sum(dim=1, keepdim=True) - betas)

        return pred

class DINA_Decoder(nn.Module):
    def __init__(self):
        pass

    def __init__(self):
        pass

    def forward(self):
        pass


class Trans_Decoder(nn.Module):
    """
        A deep neural network to decode interactions from latent knowledge proficiency.
    """

    def __init__(self, knowledge_num, exer_num, type='IRT'):
        super(Trans_Decoder, self).__init__()
        self.knowledge_num = knowledge_num
        self.exer_num = exer_num
        self.type = type
        if self.type == 'IRT':
            self.theta = nn.Linear(knowledge_num, 1)
            self.a = nn.Embedding(exer_num, 1)
            self.b = nn.Embedding(exer_num, 1)
            self.c = nn.Embedding(exer_num, 1)
        elif self.type == 'NCD':
            self.embedding = nn.Embedding(exer_num, knowledge_num)
            self.net = nn.Sequential(
                nn.Linear(knowledge_num, 256),
                nn.Sigmoid(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        elif self.type == 'MIRT':
            pass

        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, batch, user, exercise):
        if self.type == 'NCD':
            # batch_size * knowledge_num
            batch = batch[user]
            exercise_emb = self.embedding(exercise)
            return self.net(batch * exercise_emb).view(batch.shape[0], -1)

        elif self.type == 'IRT':
            # batch_size * knowledge_num
            batch = batch[user]
            theta = self.theta(batch)
            b = self.b(exercise)
            a = self.a(exercise)
            c = self.c(exercise)

            return c + (1 - c) / (1 + torch.exp(-1.7 * a * (theta - b)))
