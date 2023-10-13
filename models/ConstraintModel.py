import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import logging


class ConstraintModel(nn.Module):
    def __init__(self, know_num, embed_size, kg_mat):
        super(ConstraintModel, self).__init__()
        self.kg_mat = kg_mat
        self.embed_size = embed_size
        self.know_num = know_num
        self.model = nn.Sequential(
            nn.Linear(know_num, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, know_num),
            nn.ReLU()
        )
        self.loss_func = nn.BCELoss()

        self.x_1, self.y_1 = torch.where(self.kg_mat == 1)
        self.index_1 = torch.stack([self.x_1, self.y_1], dim=-1)

        self.index_0 = []
        self.x_0, self.y_0 = torch.where(self.kg_mat == 0)
        for x_0, y_0 in zip(self.x_0, self.y_0):
            if not x_0 == y_0:
                self.index_0.append([x_0, y_0])
        self.index_0 = torch.IntTensor(self.index_0).to(self.index_1.device)

        print("Has relation", self.index_1.shape[0])
        print("None relation", self.index_0.shape[0])

    def forward(self, x_delta):
        # x_delta : batch_size * know_num
        x_delta = self.model(x_delta)
        x_delta = x_delta.unsqueeze(-1)  # batch_size * know_num * 1
        x_delta_T = x_delta.permute(0, 2, 1)  # batch_size * 1 * know_num
        kg_mat_pred = torch.matmul(x_delta, x_delta_T)  # batch_size * know_num * know_num
        kg_mat_pred = torch.sum(kg_mat_pred, dim=0)  # know_num * know_num
        kg_mat_pred = torch.sigmoid(kg_mat_pred)

        sampled_zero = torch.randint(low=0, high=self.index_0.shape[0], size=(self.index_1.shape[0], ))
        target_0 = self.index_0[sampled_zero] # sampled_size * 2
        target_1 = self.index_1 # sampled_size * 2
        target_x = torch.cat([
            target_0[:, 0],
            target_1[:, 0]
        ])
        target_y = torch.cat([
            target_0[:, 1],
            target_1[:, 1]
        ])
        target = self.kg_mat[target_x, target_y]
        pred = kg_mat_pred[target_x, target_y]
        return self.loss_func(pred, target)
