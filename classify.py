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


from torch.utils.data import Dataset, DataLoader

class impute_set(Dataset):
    def __init__(self, x, y, t_step=48, feature=35):
        super(impute_set, self).__init__()
        print("x shape: ", x.shape)
        print("y shape: ", y.shape)
        self.x = x.reshape(-1, 48, 35)
        self.y = y.reshape(-1, 2)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]




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
        self.output_linear = nn.Linear(hidden_num, 1)
        self.softmax =nn.Softmax(dim=2)
        self.hidden = None


    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        y = self.output_linear(self.hidden)
        y = torch.sigmoid(y)
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
    print("positive rate: ", np.sum(y) / B)
    cls_model = svm.SVC(kernel='rbf', decision_function_shape='ovo', probability=True, max_iter=10000)
    cls_model.fit(x, y)
    # test
    print("start test svm")
    B, L, N = test_x.shape
    x = test_x.reshape(B, -1).cpu().numpy()
    y = test_y[:, 1].cpu().numpy()
    print("positive rate: ", np.sum(y) / B)
    probs = cls_model.predict_proba(x)[:, 1]
    auroc = roc_auc_score(y, probs)
    print(" AUROC: {:.4f}".format(auroc))

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


def GRU_Classifier(train_set, test_set, mode="train", model=None):
    # model = GRUModel(N, 128, 2).to("cuda:0")
    the_best_auc_roc = 0
    train_iter = DataLoader(dataset=train_set, \
                           batch_size=64, \
                           shuffle=True, \
                           )
    test_iter = DataLoader(dataset=test_set, \
                           batch_size=test_set.__len__(), \
                           shuffle=True, \
                           )

    for epoch in range(200):
        BCEloss = torch.nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        average_loss = 0
        for train_x, train_y in train_iter:
            train_y = train_y.to(train_x.device)
            optim.zero_grad()
            pred = model(train_x)
            pred = pred.squeeze(0).squeeze(-1)
            loss = BCEloss(pred, train_y[:, 1])
            average_loss = average_loss + loss.item()
            loss.backward()
            optim.step()
            average_loss = average_loss + loss.item()

        model.eval()
        with torch.no_grad():
            for test_x, test_y in test_iter:
                test_y = test_y.to(test_x.device)
                pred = model(test_x)
                pred = pred.squeeze(0)
                auroc = roc_auc_score(test_y.cpu().numpy()[:, 1], pred.cpu().numpy())
                if the_best_auc_roc < auroc:
                    the_best_auc_roc = auroc
        print(" AUROC(GRU): {:.4f}".format(auroc))



def classify(classifier="gru", model_path="", path="", device=""):



    train_loader = data_loader.get_loader(batch_size = 64, type="train")
    test_loader = data_loader.get_loader(batch_size = 64, type="test")

    N = 35
    model = GRUModel(N, 128, 2).to(device)
    train_set_x = []
    train_set_y = []

    imputer = brits.Model(64, 1.0, 0.0).to(device)
    imputer.load_state_dict(torch.load(model_path + '/' + "model_brits.pth"))
    print("impute train set: ", len(train_loader))
    batch_no = 0
    for train_batch in train_loader:
        with torch.no_grad():
            data = collate_fn(train_batch)
            ret = imputer.run_on_batch(data, None)
            train_x = ret["imputations"]
            cond_masks = train_batch['masks'].to(train_x.device)
            eval_masks = train_batch['eval_masks'].to(train_x.device)
            cond_obs = train_batch['values'].to(train_x.device)
            train_x = (1-cond_masks) * train_x + cond_masks * cond_obs
            train_y = train_batch["y"]
            train_set_x.append(train_x)
            train_set_y.append(train_y)
            batch_no = batch_no + 1
            print("batch no: ", batch_no)
    # if classifier == "gru":
    #     GRU_Classifier(train_set_x, train_set_y, None, None, mode="train", model=model)

    print("impute test set: ", len(test_loader))
    batch_no = 0
    test_set_x = []
    test_set_y = []
    for test_batch in test_loader:
        with torch.no_grad():
            data = collate_fn(test_batch)
            ret = imputer.run_on_batch(data, None)
            test_x = ret["imputations"]
            cond_masks = test_batch['masks'].to(test_x.device)
            eval_masks = test_batch['eval_masks'].to(train_x.device)
            cond_obs = test_batch['values'].to(test_x.device)
            test_x = (1-cond_masks) * test_x + cond_masks * cond_obs
            test_y = test_batch["y"]
            batch_no =  batch_no + 1
            print("batch no: ", batch_no)
            test_set_x.append(test_x)
            test_set_y.append(test_y)

    if classifier == "gru":
        test_set_x = torch.cat(test_set_x, dim=0)
        test_set_y = torch.cat(test_set_y, dim=0)
        train_set_x = torch.cat(train_set_x, dim=0)
        train_set_y = torch.cat(train_set_y, dim=0)
        train_set = impute_set(train_set_x,train_set_y)
        test_set = impute_set(test_set_x,test_set_y)

        GRU_Classifier(train_set, test_set, model=model)

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
    classify(classifier="lr", model_path="/home/comp/csjwxu/BRITS", path="/home/comp/csjwxu/CSDI/save/physio_fold0_20220311_151849/config.json", device="cuda:0")
