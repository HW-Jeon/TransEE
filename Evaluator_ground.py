import torch

import config_TransEE as cfgs
import utilities as util
from openke.config import Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model.TransEE import TransEE

if __name__ == "__main__":
    # print(util.get_csv_path_short())
    # input()

    util.endl(f"[Mode]          - {cfgs.MODE_EVALUATION}")
    util.endl(f"[MIN/MAX]       - {cfgs.entropy_normal_min} / {cfgs.entropy_normal_max}")
    util.endl(f"[NORM MODE]     - {cfgs.MODE_EVAL_NORM}")

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    # print(test_dataloader.testTotal)
    # input()

    # define the model
    transee = TransEE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
    )

    # for path_id in util.get_csv_path():
    #     cfgs.entropy_path_id = path_id

    # cfgs.WRITE_EVAL_RESULT = True

    for ths in cfgs.num_count_thresholds:
        cfgs.num_count_threshold = ths

        for path_id in util.get_csv_path_short():
            cfgs.entropy_path_id_short = path_id

            for type_entropy in cfgs.column_List_entropy_NM:
                cfgs.GROUND_SCORE = {
                    "TransE": 0,
                    "TransH": 0,
                    "TransR": 0,
                    "TransD": 0,
                }

                cfgs.GROUND_RANK = {
                    "TransE": 0,
                    "TransH": 0,
                    "TransR": 0,
                    "TransD": 0,
                }

                cfgs.GROUND_RANK_DOWN = {
                    "TransE": 0,
                    "TransH": 0,
                    "TransR": 0,
                    "TransD": 0,
                }

                cfgs.types_of_entropy = type_entropy

                if "diff" in type_entropy:
                    cfgs.reverse_flag = True

                else:
                    cfgs.reverse_flag = False

                if cfgs.WRITE_EVAL_RESULT:
                    util.endl(f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}] - EVAL_RESULT Writing")

                else:
                    util.endl(f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}]")

                tester = Tester(model=transee, data_loader=test_dataloader, use_gpu=True)
                tester.run_link_prediction(type_constrain=False)

                util.endl("GROUND_SCORE")

                print(cfgs.GROUND_SCORE)

                util.endl("GROUND_RANK")

                print(cfgs.GROUND_RANK)
                print(cfgs.GROUND_RANK_DOWN)

                util.endl("EVALUATION DOME")

                del tester
                torch.cuda.empty_cache()

                if cfgs.num_count_threshold < 0:
                    break
            if cfgs.num_count_threshold < 0:
                break

            cfgs.WRITE_EVAL_RESULT = False
