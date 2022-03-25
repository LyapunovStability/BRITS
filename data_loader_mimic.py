import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MySet(Dataset):
    def __init__(self, type):
        super(MySet, self).__init__()
        missing_rate = 0.4
        print("missing rate: ", missing_rate)
        data = torch.load("../DATA/mimic/mimic_{0}_{1}.pt".format(type, missing_rate))
        self.X = data['X_scalar'].float().cpu()
        self.M = data['M'].float().cpu()
        self.y = data['Y'].float().cpu()
        self.label = data['label'].float().cpu()
        self.label_mask = data['label_mask'].float().cpu()


        self.X_b = self.X.numpy()
        self.X_b = torch.from_numpy(np.flip(self.X_b, axis=1).copy())
        self.delta = self.Timelag_gen(self.M, direction="forward")
        self.delta_b = self.Timelag_gen(self.M, direction="backward")
        self.M_b = self.M.numpy()
        self.M_b = torch.from_numpy(np.flip(self.M_b, axis=1).copy())
        self.label_mask_b = self.label_mask.numpy()
        self.label_mask_b = torch.from_numpy(np.flip(self.label_mask_b, axis=1).copy())

        self.train_std = 1
        self.train_mean = 0


        self.type = type

    def Timelag_gen(self, M, direction="backward"):
        B = M.shape[0]
        L = M.shape[1]
        N = M.shape[2]

        delta = torch.zeros((B, L, N)).to(M.device)
        steps = L
        for i in range(B):
            if direction == "backward":
                for k in range(L-1, -1, -1):
                    if k == L-1:
                        delta[i, k, :] = 1
                    else:
                        delta[i, k, :] = 1 + (1 - M[i, k, :]) * delta[i, k + 1, :]
            else:
                for k in range(L):
                    if k == 0:
                        delta[i, k, :] = 1
                    else:
                        delta[i, k, :] = 1 + (1 - M[i, k, :]) * delta[i, k - 1, :]

        return delta



    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):


        rec = {
            "values": self.X[idx] - self.X[idx] * self.label_mask[idx],
            "evals": self.X[idx] * self.label_mask[idx],
            "masks": self.M[idx] - self.label_mask[idx],
            "eval_masks": self.label_mask[idx],
            "forwards": torch.arange(0, 48),
            "deltas": self.delta[idx],
            "values_backward": self.X_b[idx] - self.X_b[idx] * self.label_mask_b[idx],
            "evals_backward": self.X_b[idx] * self.label_mask_b[idx],
            "masks_backward": self.M_b[idx] - self.label_mask_b[idx],
            "eval_masks_backward": self.label_mask_b[idx],
            "forwards_backward": torch.arange(0, 48),
            "deltas_backward": self.delta_b[idx],
            "y": self.y[idx]

        }

        if self.type == "train":
            rec['is_train'] = 1
        else:
            rec['is_train'] = 0
        if int(idx//2) == 0:
            rec['labels'] = 1
        else:
            rec['labels'] = 0
        return rec




def get_loader(batch_size=64, shuffle=True, type="train"):
    data_set = MySet(type)
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=4, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           )

    return data_iter