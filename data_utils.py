import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
import json
from torch.utils.data import Dataset


def meta_mlp_cog_data_load(meta_ratio, noise_ratio, train_path, valid_path, test_path, concept_path, kg_path, dataset_info):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
    kg_mat = np.load(kg_path, allow_pickle=True)

    n_user = dataset_info['stu_all']
    n_item = dataset_info['exer_all']
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')


    np.random.shuffle(train_list)
    meta_len = int(len(train_list) * meta_ratio)
    meta_list = train_list[:meta_len]
    noised_list = train_list[meta_len:]
    noise_ratio = (len(train_list) * noise_ratio) / len(noised_list)

    if noise_ratio > 0:
        for i in range(len(noised_list)):
            if np.random.random() < noise_ratio:
                noised_list[i, 2] = 1 - noised_list[i, 2]  # 0 和 1 翻转
    train_data = sp.coo_matrix(([1 if i == 1 else -1 for i in noised_list[:, 2]],
                                (noised_list[:, 0], noised_list[:, 1])), dtype='float64', shape=(n_user, n_item))

    with open(concept_path, mode="r") as F:
        concept_map = json.load(F)
    concept_map = [i[1] for i in sorted(list(concept_map.items()), key=lambda x: int(x[0]))]

    return meta_list, np.concatenate((meta_list, noised_list), axis=0), train_data.A, valid_list, test_list, concept_map, kg_mat


class DataCog(Dataset):
    def __init__(self, train_list, concept_map, know_num):
        self.concept_map = concept_map
        self.know_num = know_num
        self.data = train_list
        self.knowledge = self.convert_triple_know(concept_map, know_num)

    def __getitem__(self, index):
        triple = self.data[index]
        return triple, self.knowledge[triple[1]]

    def __len__(self):
        return len(self.data)

    def convert_triple_know(self, concept_map, know_num):
        knowledge = np.zeros(shape=(len(concept_map), know_num), dtype=np.float32)
        for i, ii in enumerate(concept_map):
            knowledge[i, ii] = 1
        return knowledge


class DataDiffusion(Dataset):
    def __init__(self, data_0, data_1, concept_map, know_num):
        self.concept_map = concept_map
        self.know_num = know_num
        self.data_0, self.target_index_0 = self.convert_item_seq2matrix(data_0)
        self.data_1, self.target_index_1 = self.convert_item_seq2matrix(data_1)

    def __getitem__(self, index):
        return self.data_0[index], self.convert_item_know(self.data_0[index]), self.target_index_0[index], \
               self.data_1[index], self.convert_item_know(self.data_1[index]), self.target_index_1[index]

    def __len__(self):
        return len(self.data_0)

    def convert_item_know(self, data):
        knowledge = np.zeros(shape=(len(data), self.know_num), dtype=np.int32)
        for i, ii in enumerate(data):
            v = self.concept_map[ii]
            knowledge[i, v] = 1
        return knowledge

    def convert_item_seq2matrix(self, item_dict):
        max_length = max([len(item) for item in item_dict.values()])
        matrix = np.zeros((len(item_dict), max_length), dtype=np.int32)
        # know_matrix = np.zeros((len(item_dict), max_length, self.know_num), dtype=np.int32)
        for key, value in item_dict.items():
            for y, yy in enumerate(value):
                matrix[key, y] = yy
                # v = self.concept_map[yy]
                # know_matrix[key, y, v] = 1
        target_index = [len(i) - 1 for i in item_dict.values()]
        return matrix, target_index
