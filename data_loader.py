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
        self.content = open('./json/{}.json'.format(type)).readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())
        self.type = type
    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if self.type == "train":
            rec['is_train'] = 1
        else:
            rec['is_train'] = 0
        
        return rec

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))
    
    def to_tensor_dict(recs):

        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))
        return {
            'values': values,
            'forwards': forwards,
            'masks': masks,
            'deltas': deltas,
            'evals': evals,
            'eval_masks': eval_masks
        }
        return {
            'values': values.permute(0,2,1),
            'forwards': forwards.permute(0,2,1),
            'masks': masks.permute(0,2,1),
            'deltas': deltas.permute(0,2,1),
            'evals': evals.permute(0,2,1),
            'eval_masks': eval_masks.permute(0,2,1)
        }
    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}
    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))
    
    return ret_dict



def get_loader(batch_size = 64, shuffle = True, type="train"):
    data_set = MySet(type)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
