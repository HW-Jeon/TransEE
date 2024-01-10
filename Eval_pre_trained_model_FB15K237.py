import torch

import config_TransEE as cfgs
import openke
import utilities as util
from openke.config import Tester, Trainer
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model import TransD, TransE, TransH, TransR
from openke.module.strategy import NegativeSampling

if __name__ == "__main__":
    cfgs.dataset = "FB15K237"
    cfgs.data_tag = "/Pre_FB15K237"
    cfgs.num_count_threshold = -1
    cfgs.EVAL_DEFALT_TF = 10

    # print(util.get_csv_path_short())
    util.print_entropy_header()

    models = util.load_models_list(tags="/Pre_FB15K237", devices=cfgs.devices)
    # models = util.load_models_list(devices="cpu")

    # models, dataloaders = util.load_models_loader(tags="/0925", dataset="FB15K237", devices="cpu")

    # print("Loading models Done")

    for path_id in util.get_csv_path_short():
        cfgs.entropy_path_id_short = path_id

        for strModel in cfgs.strModels:
            print(strModel)
            test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

            tester = Tester(model=models[strModel], data_loader=test_dataloader, use_gpu=True)
            tester.run_link_prediction(type_constrain=False)
