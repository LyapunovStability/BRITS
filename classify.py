import torch
import torch.nn as nn
import json
import yaml
from sklearn import svm
import torch.nn.functional as F
import numpy as np
from torch import optim
import math
import time
from math import sqrt
from models import brits
import data_loader_physio as data_loader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def collate_fn(recs):
    def to_tensor_dict(recs, direction="forward"):
        if direction == "forward":
            type = ""
        else:
            type = "_backward"

        return {
            'values': recs["values" + type].cuda(),
            'forwards': recs["forwards" + type].cuda(),
            'masks': recs["masks" + type].cuda(),
            'deltas': recs["deltas" + type].cuda(),
            'evals': recs["evals" + type].cuda(),
            'eval_masks': recs["eval_masks" + type].cuda()
        }

    ret_dict = {'forward': to_tensor_dict(recs, direction="forward"),
                'backward': to_tensor_dict(recs, direction="backward")}
    ret_dict['is_train'] = recs["is_train"].cuda()
    ret_dict['labels'] = recs['labels'].cuda()
    return ret_dict




class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.softmax =nn.Softmax(dim=2)
        self.hidden = None


    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        y = self.output_linear(self.hidden)
        y = self.softmax(y)
        return y


class GRURegressor(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(GRURegressor, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.softmax =nn.Softmax(dim=2)
        self.hidden = None


    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        y = self.output_linear(self.hidden)
        return y

def SVM(train_x, train_y, test_x, test_y):
    B, L, N = train_x.shape
    # train
    print("start train svm")
    x = train_x.reshape(B, -1).cpu().numpy()
    y = train_y[:, 1].cpu().numpy()
    cls_model = svm.SVC(kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True, max_iter=10000)
    cls_model.fit(x, y)
    # test
    print("start test svm")
    B, L, N = test_x.shape
    x = test_x.reshape(B, -1).cpu().numpy()
    y = test_y[:, 1].cpu().numpy()
    probs = cls_model.predict_proba(x)[:, 1]
    auroc = roc_auc_score(y, probs)
    print(" AUROC(model:logit): {:.4f}".format(auroc))

def Logistic_Regression(train_x, train_y, test_x, test_y):
    B, L, N = train_x.shape
    # train
    print("start train lr")
    x = train_x.reshape(B, -1).cpu().numpy()
    y = train_y[:, 1].cpu().numpy()
    cls_model = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
    cls_model.fit(x, y)
    # test
    print("start test lr")
    B, L, N = test_x.shape
    x = test_x.reshape(B, -1).cpu().numpy()
    y = test_y[:, 1].cpu().numpy()
    probs = cls_model.predict_proba(x)[:, 1]
    auroc = roc_auc_score(y, probs)
    print(" AUROC(model:logit): {:.4f}".format(auroc))


def GRU_Classifier(train_x, train_y, test_x, test_y, mode="train", model=None):

    # model = GRUModel(N, 128, 2).to("cuda:0")
    if mode == "train":
        B, L, N = train_x.shape
        BCEloss = torch.nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.005)
        for epoch in range(100):
            model.train()
            optim.zero_grad()
            pred = model(train_x)
            loss = BCEloss(pred.squeeze(0), train_y)
            loss.backward()
            optim.step()
    else:
        with torch.no_grad():
            pred = model(test_x)
            pred = pred.squeeze(0)
            auroc = roc_auc_score(test_y.cpu().numpy(), pred.cpu().numpy())
            print("AUROC(model:GRU): {:.4f}".format(auroc))

def classify(classifier="gru", model_path="", path="", device=""):

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))

    train_loader = data_loader.get_loader(batch_size = 64, type="train")
    test_loader = data_loader.get_loader(batch_size = 64, type="test")

    N = 35
    model = GRUModel(N, 128, 2).to("cuda:0")
    train_set_x = []
    train_set_y = []

    imputer = brits(config, device=device).to(device)
    imputer.load_state_dict(torch.load(model_path + '/' + "model.pth"))
    print("impute train set: ", len(train_loader))
    batch_no = 0
    for train_batch in train_loader:
        with torch.no_grad():
            data = collate_fn(train_batch)
            ret = imputer.run_on_batch(data, None)
            train_x = ret["imputation"]
            train_y = train_batch["y"]
            train_set_x.append(train_x)
            train_set_y.append(train_y)
            batch_no = batch_no + 1
            print("batch no: ", batch_no)
        if classifier == "gru":
            GRU_Classifier(train_x, train_y, None, None, mode="train", model=model)

    print("impute test set: ", len(test_loader))
    batch_no = 0
    test_set_x = []
    test_set_y = []
    for test_batch in test_loader:
        with torch.no_grad():
            data = collate_fn(train_batch)
            ret = imputer.run_on_batch(data, None)
            test_x = ret["imputation"]
            test_y = test_batch["y"]
            batch_no =  batch_no + 1
            print("batch no: ", batch_no)
            test_set_x.append(test_x)
            test_set_y.append(test_y)
        if classifier == "gru":
            GRU_Classifier(None, None, test_x, test_y, mode="test", model=model)

    if classifier == "svm" or classifier == "lr":
        train_set_x = torch.cat(train_set_x, dim=0)
        train_set_y = torch.cat(train_set_y, dim=0)
        test_set_x = torch.cat(test_set_x, dim=0)
        test_set_y = torch.cat(test_set_y, dim=0)
        if classifier == "lr":
            Logistic_Regression(train_set_x, train_set_y, test_set_x, test_set_y)
        elif classifier == "svm":
            SVM(train_set_x, train_set_y, test_set_x, test_set_y)


if __name__ == '__main__':
    classify(classifier="lr", model_path="/home/comp/csjwxu/CSDI/save/physio_fold0_20220311_151849", path="/home/comp/csjwxu/CSDI/save/physio_fold0_20220311_151849/config.json", device="cuda:0")
