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

    # cfgs.default_entropy_dir_path = "./csv/FB15K237/GOOD_PERFOMANCE/PDF_Categorical_Mixed/entropy_k_"
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_05_075/entropy_k_"
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_025_05/entropy_k_"
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_0.75_0.5/entropy_k_"

    # cfgs.setDataset("WN18RR")
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_TRUE_Mixed_0.5_0.75/entropy_k_"

    strDataset = "FB15K237"
    cfgs.default_entropy_dir_path = (
        # "./csv/FB15K237/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        "./csv/FB15K237/PDF_Categorical_TRUE_Mixed_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/PER_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/TOT_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/FINAL_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/FINAL_NP_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/FINAL_PDF_PAIRED_Trained_0.5_0.75/entropy_k_"
    )

    # strDataset = "WN18RR"
    # cfgs.default_entropy_dir_path = (
    #     # f"./csv/{strDataset}/DEFAULT_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    #     f"./csv/{strDataset}/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    # )

    cfgs.setDataset(strDataset)

    if "WN" in strDataset:
        cfgs.hit_k_limits = [3]

    # /home/kist/workspace/OpenKE/csv/FB15K/DEFAULT_RES_PDF_Categorical_Trained_0.5_0.75
    # # dataloader for training
    # train_dataloader = TrainDataLoader(
    #     in_path="./benchmarks/FB15K237/",
    #     nbatches=100,
    #     threads=8,
    #     sampling_mode="normal",
    #     bern_flag=1,
    #     filter_flag=1,
    #     neg_ent=25,
    #     neg_rel=0,
    # )

    # # dataloader for test
    # test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
    train_dataloader = util.dataLoader(strDataset)

    test_dataloader = TestDataLoader(f"./benchmarks/{strDataset}/", "link")

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

    # for eval_norm_mode in ["LOGIT"]:
    for eval_norm_mode in ["MINMAX", "LOGIT"]:
        cfgs.MODE_EVAL_NORM = eval_norm_mode

        for ths in cfgs.num_count_thresholds:
            cfgs.num_count_threshold = ths

            for path_id in util.get_csv_path_short():
                cfgs.entropy_path_id_short = path_id

                for type_entropy in cfgs.column_List_entropy_MINIMUM:
                    # for type_entropy in ["num_ori"]:

                    cfgs.reverse_flag = False
                    # cfgs.reverse_flag = True

                    if eval_norm_mode == "MINMAX":
                        cfgs.entropy_normal_min = 3.125

                        # if "c_renyi" in type_entropy:
                        #     # cfgs.entropy_normal_min = 3.6875
                        #     cfgs.entropy_normal_min = 4.4531

                        # elif "n_shannon" in type_entropy:
                        #     cfgs.entropy_normal_min = 2.1875
                        #     # cfgs.entropy_normal_min = 4.375

                    else:
                        if "c_renyi" in type_entropy:
                            cfgs.entropy_normal_min = 3.6875
                        # cfgs.entropy_normal_min = 4.4531

                        elif "n_shannon" in type_entropy:
                            # cfgs.entropy_normal_min = 2.1875
                            cfgs.entropy_normal_min = 4.375

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

                    util.print_eval_header()

                    # if "diff" in type_entropy:
                    #     cfgs.reverse_flag = True

                    # else:
                    #     cfgs.reverse_flag = False

                    if cfgs.WRITE_EVAL_RESULT:
                        util.endl(
                            f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}] - EVAL_RESULT Writing"
                        )

                    else:
                        util.endl(
                            f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}]"
                        )

                    tester = Tester(
                        model=transee, data_loader=test_dataloader, use_gpu=True
                    )
                    tester.run_link_prediction(type_constrain=False)
                    # tester.run_triple_classification()

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
