# coding:utf-8
import copy
import ctypes
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from tqdm import tqdm

import config_TransEE as cfgs
import utilities as util


class Tester(object):
    def __init__(self, model=None, data_loader=None, use_gpu=True, pre_train=False):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

        if pre_train:
            cfgs.MODE_EVALUATION = "DEFAULT"
        else:
            self.setEntropy_from_csv()

    def setEntropy_from_csv(self):
        self.entropy_df = {}
        self.paths = util.get_csv_path()
        for path in self.paths:
            self.entropy_df[path] = pd.read_csv(path, dtype=cfgs.entropy_column_dtypes)

    def tf_entropy_score(self, r, mode, path=None):
        if cfgs.num_count_threshold == 0:
            return None

        if cfgs.num_count_threshold < 0:
            tf = cfgs.EVAL_DEFALT_TF

        else:
            tf = cfgs.num_count_threshold

        path = cfgs.entropy_path_id_short.replace("Tag", mode)

        _data = self.entropy_df[path][self.entropy_df[path]["relation"] == r]

        data = _data[_data["num_ori"] >= tf]

        # print(len(data))
        # print(len(cfgs.strModels))

        if len(data) == 0:
            # print(mode, r, len(data), len(cfgs.strModels))
            return False

        return True

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        return self.model.predict(
            {
                "batch_h": self.to_var(data["batch_h"], self.use_gpu),
                "batch_t": self.to_var(data["batch_t"], self.use_gpu),
                "batch_r": self.to_var(data["batch_r"], self.use_gpu),
                "mode": data["mode"],
            }
        )

    def run_link_prediction(self, type_constrain=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode("link")
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)

        cnt_head = 0
        cnt_tail = 0

        for index, [data_head, data_tail] in enumerate(training_range):
            # data_head: prediction 'HEAD' using 'r', 't'
            # data_tail: prediction 'TAIL' using 'h', 'r'

            # EXAMPLE:
            # {'batch_h': array([    0,     1,     2, ..., 14538, 14539, 14540]), 'batch_t': array([4494]), 'batch_r': array([0]), 'mode': 'head_batch'}
            # {'batch_h': array([27]), 'batch_t': array([    0,     1,     2, ..., 14538, 14539, 14540]), 'batch_r': array([0]), 'mode': 'tail_batch'}

            training_range.set_description(str(data_head["batch_r"][0]))
            cfgs.CURRENT_REL = data_head["batch_r"][0]
            cfgs.ground["head_batch"] = data_tail["batch_h"][0]
            cfgs.ground["tail_batch"] = data_head["batch_t"][0]

            if "DROP_TRUE" in cfgs.MODE_EVALUATION:
                if self.tf_entropy_score(data_head["batch_r"][0], "head_batch"):
                    score = self.test_one_step(data_head)
                    self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
                    cnt_head += 1
                if self.tf_entropy_score(data_head["batch_r"][0], "tail_batch"):
                    score = self.test_one_step(data_tail)
                    self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
                    cnt_tail += 1

            else:
                score = self.test_one_step(data_head)
                self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
                score = self.test_one_step(data_tail)
                self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)

        if "DROP_TRUE" in cfgs.MODE_EVALUATION:
            self.lib.test_link_prediction(type_constrain, cnt_head, cnt_tail)
        else:
            self.lib.test_link_prediction(type_constrain, 0, 0)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print(hit10)
        return mrr, mr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1, 1), score.reshape(-1, 1)], axis=-1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod=None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode("classification")
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis=-1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1, 1), score.reshape(-1, 1)], axis=-1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
