import numpy as np
import torch

import config_TransEE as cfgs
import utilities as util
from openke.config import Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model.TransEE import TransEE


def link_predictions(models, test_dataloader, mins):
    cfgs.entropy_normal_min = mins
    util.endl(f"[min/max] {cfgs.entropy_normal_min} {cfgs.entropy_normal_max}")

    tester = Tester(model=models, data_loader=test_dataloader, use_gpu=True)
    _, _, hit10, _, _ = tester.run_link_prediction(type_constrain=False)

    del tester
    torch.cuda.empty_cache()

    return hit10


def hpo_num_threshold(models, test_dataloader):
    result = {}

    for i in range(5, 51):
        if "GROUND" in cfgs.MODE_HPO:
            cfgs.EVAL_DEFALT_TF = i
        else:
            cfgs.num_count_threshold = i

        util.endl(f"[{i} - {path_id} {type_entropy} {cfgs.reverse_flag}]")

        tester = Tester(model=models, data_loader=test_dataloader, use_gpu=True)
        _, _, hit10, _, _ = tester.run_link_prediction(type_constrain=False)

        result[i] = hit10

        del tester
        torch.cuda.empty_cache()

    print(result)


def adjust_config(result_eval):
    """Adjust the configuration based on the result evaluation."""
    if result_eval > 0:
        cfgs.entropy_normal_min += cfgs.entropy_normal_min / 2
    elif result_eval < 0:
        cfgs.entropy_normal_min /= 2


def recursive_evaluations(
    models,
    test_dataloader,
    old_hit10,
    new_min,
    old_min,
    neg_count=0,
):
    print("\nneg_count: ", neg_count)
    if new_min > 5 or new_min < 0.5:
        return 0.0, new_min

    weight = new_min * 0.5 if old_hit10 is None else abs(old_min - new_min) / 2

    hit10 = link_predictions(models, test_dataloader, new_min)

    if old_hit10 is not None:
        if old_hit10 - hit10 < 0.0005:
            neg_count += 1

        else:
            neg_count -= 1

        if neg_count >= 4 or weight <= 0.0005 or new_min > 5 or new_min < 0.5:
            return hit10, new_min

    left, left_val = recursive_evaluations(
        models, test_dataloader, hit10, round(new_min - weight, 4), new_min, neg_count
    )
    right, right_val = recursive_evaluations(
        models, test_dataloader, hit10, round(new_min + weight, 4), new_min, neg_count
    )

    if hit10 == max(hit10, left, right):
        return (hit10, new_min)

    return (left, left_val) if left > right else (right, right_val)


def set_FB15K237():
    cfgs.setDataset("FB15K237")

    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_05_075/entropy_k_"
    cfgs.default_entropy_dir_path = (
        "./csv/FB15K237/PDF_Categorical_TRUE_Trained_0.5_0.75/entropy_k_"
    )


def set_FB15K():
    cfgs.setDataset("FB15K")
    cfgs.default_entropy_dir_path = (
        "./csv/FB15K/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    )


def set_WN18RR():
    cfgs.setDataset("WN18RR")
    cfgs.default_entropy_dir_path = (
        "./csv/WN18RR/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    )


def setConfig(strDataset):
    cfgs.setDataset(strDataset)
    cfgs.default_entropy_dir_path = (
        # f"./csv/{strDataset}/DEFAULT_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        # f"./csv/{strDataset}/PER_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        # f"./csv/{strDataset}/AVG_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        # f"./csv/{strDataset}/PER_PDF_Trained_0.5_0.75/entropy_k_"
        # f"./csv/{strDataset}/FINAL_PDF_Trained_0.5_0.75/entropy_k_"
        # f"./csv/{strDataset}/FINAL_PDF_PAIRED_Trained_0.5_0.75/entropy_k_"
        f"./csv/{strDataset}/FINAL_N_PDF_Trained_0.5_0.75/entropy_k_"
    )


if __name__ == "__main__":
    # print(util.get_csv_path_short())
    # input()

    # util.endl(
    #     f"\n[EVAL MODE]         - {cfgs.MODE_EVALUATION}"
    #     + f"\n[HPO  MODE]         - {cfgs.MODE_HPO}"
    #     + f"\n[NORM MODE]         - {cfgs.MODE_EVAL_NORM}\n"
    # )
    # util.endl(f"\n[HPO MODE]     - {cfgs.MODE_EVAL_NORM}")
    # # util.endl(f"[MIN/MAX]       - {cfgs.entropy_normal_min} / {cfgs.entropy_normal_max}")
    # util.endl(f"\n[NORM MODE]     - {cfgs.MODE_EVAL_NORM}")

    # set_FB15K237()

    # set_WN18RR()

    # strDS = "FB15K"
    strDS = "FB15K237"
    # strDS = "WN18RR"

    # cfgs.MODE_EVALUATION = "DROP_TRUE"
    cfgs.MODE_EVALUATION = "DEFAULT"

    # dataloader for training

    # if "237" in strDS:
    #     set_FB15K237()

    # else:
    setConfig(strDS)

    if "WN" in strDS:
        cfgs.hit_k_limits = [3]

    # dataloader for test
    train_dataloader = util.dataLoader(strDS)

    # dataloader for test
    test_dataloader = TestDataLoader(f"./benchmarks/{strDS}/", "link")

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
    for eval_norm_mode in ["MINMAX", "LOGIT"]:
        cfgs.MODE_EVAL_NORM = eval_norm_mode

        for ths in cfgs.num_count_thresholds:
            cfgs.num_count_threshold = ths

            for path_id in util.get_csv_path_short():
                cfgs.entropy_path_id_short = path_id

                for type_entropy in cfgs.column_List_entropy_MINIMUM:
                    if "MINNORM" not in cfgs.MODE_HPO:
                        if eval_norm_mode == "MINMAX":
                            if "c_renyi" in type_entropy:
                                # cfgs.entropy_normal_min = 3.6875
                                cfgs.entropy_normal_min = 4.4531

                            elif "n_shannon" in type_entropy:
                                # cfgs.entropy_normal_min = 2.1875
                                cfgs.entropy_normal_min = 4.375

                        else:
                            if "c_renyi" in type_entropy:
                                cfgs.entropy_normal_min = 3.6875
                                # cfgs.entropy_normal_min = 4.4531

                            elif "n_shannon" in type_entropy:
                                cfgs.entropy_normal_min = 2.1875
                                # cfgs.entropy_normal_min = 4.375

                    else:
                        cfgs.entropy_normal_min = 5.0

                    cfgs.reset_GROUND()

                    cfgs.types_of_entropy = type_entropy

                    util.print_eval_header()

                    if "diff" in type_entropy:
                        cfgs.reverse_flag = True

                    else:
                        cfgs.reverse_flag = False

                    if "THRESHOLD" in cfgs.MODE_HPO:
                        hpo_num_threshold(transee, test_dataloader)
                    elif "MINNORM" in cfgs.MODE_HPO:
                        util.endl(
                            f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}]"
                        )
                        print(f"--------[ Fast Mode ] : {cfgs.MODE_HPO_FAST} ")

                        hit10, val = recursive_evaluations(
                            transee,
                            test_dataloader,
                            None,
                            cfgs.entropy_normal_min,
                            None,
                        )

                        print(
                            f"\n--------[hpo Done] - {ths} - {path_id} {type_entropy} {cfgs.reverse_flag}"
                        )
                        print(f"--------[ Result ] - MIN: {val}, HIT@10: {hit10} \n")
                        # # Go Left
                    # run_evaluations(transee, test_dataloader, 0.0, "LEFT")

                    # # Go Right
                    # run_evaluations(transee, test_dataloader, 100.0, "RIGHT")

                    if cfgs.num_count_threshold < 0:
                        break
                if cfgs.num_count_threshold < 0:
                    break

                cfgs.WRITE_EVAL_RESULT = False

        if "GROUND" in cfgs.MODE_HPO:
            break
