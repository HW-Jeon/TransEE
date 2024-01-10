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
    cfgs.dataset = "WN18RR"
    cfgs.data_tag = "/Pre_WN18RR"
    cfgs.num_count_threshold = -1
    cfgs.EVAL_DEFALT_TF = 18

    models = util.load_models_list(tags="/Pre_WN18RR", devices=cfgs.devices)
    # models = util.load_models_list(devices="cpu")

    # models, dataloaders = util.load_models_loader(tags="/0925", dataset="FB15K237", devices="cpu")

    # print("Loading models Done")
    cfgs.default_entropy_dir_path = (
        # f"./csv/{strDataset}/DEFAULT_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        f"./csv/WN18RR/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    )

    cfgs.hit_k_limits = [3]

    for strModel in cfgs.strModels:
        print(strModel)

        test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

        tester = Tester(
            model=models[strModel], data_loader=test_dataloader, use_gpu=True
        )
        tester.run_link_prediction(type_constrain=False)
