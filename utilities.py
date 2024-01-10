import os
import time
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special import expit, logit
from tabulate import tabulate
from torch.autograd import Variable

import config_TransEE as cfgs
from openke.data import TrainDataLoader
from openke.module.model import TransD, TransE, TransH, TransR


def endl_time(strData: str = "---------"):
    if cfgs.ctime is None:
        cfgs.ctime = time.time()

    end_time = time.time()

    if "start" not in strData:
        print(f"[{strData} time]: {end_time-cfgs.ctime:.5f}")

    cfgs.ctime = end_time


def endl(strData: str = "---------"):
    print(
        "------------------------------",
        strData,
        "-----------------------------------",
        "\n",
    )


def resize_tensor(tensor, sample_size=cfgs.MODE_MIN_RESOURCE):
    if len(tensor) > sample_size and cfgs.MODE_MIN_RESOURCE != 0:
        # Generate random unique indices
        indices = torch.randperm(len(tensor))[:sample_size]
        # Sample the tensor

        return tensor[indices]
    else:
        return tensor


def print_eval_header():
    endl(
        f"\n[ Dataset ]         - {cfgs.dataset}"
        + f"\n[EVAL MODE]         - {cfgs.MODE_EVALUATION}"
        + f"\n[HPO  MODE]         - {cfgs.MODE_HPO}"
        + f"\n[MIN / MAX]         - {cfgs.entropy_normal_min} / {cfgs.entropy_normal_max}"
        + f"\n[NORM MODE]         - {cfgs.MODE_EVAL_NORM}\n"
    )


def print_entropy_header():
    endl(
        f"\n[CURRENT_REL]         - {cfgs.CURRENT_REL}"
        + f"\n[CURRENT_HIT]         - {cfgs.CURRENT_HIT}"
        + f"\n[CURRENT_MODEL]         - {cfgs.CURRENT_MODEL}"
        + f"\n[CURRENT_BATCH]         - {cfgs.CURRENT_BATCH}"
        + f"\n[CURRENT_PAIRD]         - {cfgs.CURRENT_PAIRD}"
        + f"\n[CURRENT_LABEL]         - {cfgs.CURRENT_LABEL}\n"
    )


def endl_mode(strData: str = "---------"):
    if "DEBUG" in cfgs.MODE or "PRINTABLE" in cfgs.MODE:
        print(
            "------------------------------",
            strData,
            "-----------------------------------",
            "\n",
        )


def transe_models(tag="", strDataset="FB15K237", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True,
        devices=devices,
    )

    transe.load_checkpoint(f".{tag}/checkpoint/transe.ckpt")

    return transe, train_dataloader


def transh_models(tag, strDataset="FB15K237", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True,
        devices=devices,
    )

    transh.load_checkpoint(f".{tag}/checkpoint/transh.ckpt")

    return transh, train_dataloader


def transr_models(tag, strDataset="FB15K237", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K237/",
        nbatches=512,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=200,
        dim_r=200,
        p_norm=1,
        norm_flag=True,
        rand_init=False,
        devices=devices,
    )
    transr.load_checkpoint(f".{tag}/checkpoint/transr.ckpt")

    return transr, train_dataloader


def transd_models(tag, strDataset="FB15K237", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K237/",
        nbatches=256,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=200,
        dim_r=200,
        p_norm=1,
        norm_flag=True,
        devices=devices,
    )
    transd.load_checkpoint(f".{tag}/checkpoint/transd.ckpt")

    return transd, train_dataloader


def transe_models_WN(tag="", strDataset="WN18RR", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/WN18RR/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    transe.load_checkpoint(f".{tag}/checkpoint/transe.ckpt")

    return transe, train_dataloader


def transh_models_WN(tag, strDataset="WN18RR", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/WN18RR/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    transh.load_checkpoint(f".{tag}/checkpoint/transh.ckpt")

    return transh, train_dataloader


def transr_models_WN(tag, strDataset="WN18RR", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/WN18RR/",
        nbatches=512,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=100,
        dim_r=100,
        p_norm=1,
        norm_flag=True,
        rand_init=False,
    )
    transr.load_checkpoint(f".{tag}/checkpoint/transr.ckpt")

    return transr, train_dataloader


def transd_models_WN(tag, strDataset="WN18RR", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/WN18RR/",
        nbatches=256,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # define the model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=100,
        dim_r=100,
        p_norm=1,
        norm_flag=True,
    )
    transd.load_checkpoint(f".{tag}/checkpoint/transd.ckpt")

    return transd, train_dataloader


def transe_models_FB15K(tag="", strDataset="FB15K", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    transe.load_checkpoint(f".{tag}/checkpoint/transe.ckpt")

    return transe, train_dataloader


def transh_models_FB15K(tag, strDataset="FB15K", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    transh.load_checkpoint(f".{tag}/checkpoint/transh.ckpt")

    return transh, train_dataloader


def transr_models_FB15K(tag, strDataset="FB15K", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K/",
        nbatches=512,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=100,
        dim_r=100,
        p_norm=1,
        norm_flag=True,
        rand_init=False,
    )
    transr.load_checkpoint(f".{tag}/checkpoint/transr.ckpt")

    return transr, train_dataloader


def transd_models_FB15K(tag, strDataset="FB15K", devices=cfgs.devices):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=f"./benchmarks/FB15K/",
        nbatches=256,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=100,
        dim_r=100,
        p_norm=1,
        norm_flag=True,
    )
    transd.load_checkpoint(f".{tag}/checkpoint/transd.ckpt")

    return transd, train_dataloader


load_model = {
    "TransE": transe_models,
    "TransD": transd_models,
    "TransH": transh_models,
    "TransR": transr_models,
}

load_model_WN = {
    "TransE": transe_models_WN,
    "TransD": transd_models_WN,
    "TransH": transh_models_WN,
    "TransR": transr_models_WN,
}

load_model_FB15K = {
    "TransE": transe_models_FB15K,
    "TransD": transd_models_FB15K,
    "TransH": transh_models_FB15K,
    "TransR": transr_models_FB15K,
}

load_models = {
    "FB15K237": load_model,
    "FB15K": load_model_FB15K,
    "WN18RR": load_model_WN,
}


def load_dataset_rel(dataset_name, file_name, relation):
    triple = open(f"./benchmarks/{dataset_name}/{file_name}.txt", "r")

    head_train = np.array([])
    tail_train = np.array([])

    tot = (int)(triple.readline())
    for i in range(tot):
        content = triple.readline()
        h, t, r = content.strip().split()

        if relation == r:
            head_train = np.append(head_train, int(h))
            tail_train = np.append(tail_train, int(t))

    triple.close()

    # print(head_train)
    # print(tail_train)

    return head_train, tail_train


def to_var(x, use_gpu):
    if use_gpu:
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))


def to_var_single(x, use_gpu):
    if use_gpu:
        return Variable(torch.tensor(x).cuda())
    else:
        return Variable(torch.tensor(x))


def isSucessfulPredictions(top_k, result, options=True, column_names="tail_id"):
    strResult = str((top_k[column_names] == result).any())
    if strResult != "False":
        bool_series = top_k[column_names] == result
        index_label = bool_series.idxmax()

        # get the position of the row in the dataframe
        position = top_k.index.get_loc(index_label) + 1

        if options:
            return (
                strResult,
                position,
            )
        else:
            return position

    else:
        if options:
            return strResult
        else:
            return False


def get_min_length(mode, hit, dfs):
    (print(df[df["limit"] == hit]) for df in dfs.values())

    if mode != "mixed_batch":
        return min(
            len(df[(df["mode"] == mode) & (df["limit"] == hit)]) for df in dfs.values()
        )

    return min(len(df[df["limit"] == hit]) for df in dfs.values())


def process_df(df):
    return (
        df[["limit", "entity", "rank"]]
        .groupby(["limit", "entity"])["rank"]
        .mean()
        .reset_index()
        .sort_values(by=["rank"], ascending=True)
        .reset_index(drop=True)
    )


def load_models_list(
    strModels=cfgs.strModels, tags=cfgs.data_tag, devices=cfgs.devices
):
    print("tag: ", tags)
    print(cfgs.dataset)

    models = {}

    for strModel in strModels:
        print(strModel)

        model, _ = load_models[cfgs.dataset][strModel](tag=tags, devices=devices)

        models[strModel] = model

    return models


def load_models_loader(
    strModels=cfgs.strModels,
    tags=cfgs.data_tag,
    dataset="FB15K237",
    devices=cfgs.devices,
):
    models = {}
    dataloaders = {}

    for strModel in strModels:
        model, train_dataloader = load_model[strModel](
            tag=tags, strDataset=dataset, devices=devices
        )

        models[strModel] = model
        dataloaders[strModel] = train_dataloader

    return models, dataloaders


def dfNormalize_sigmoid_df(df_ori: pd.DataFrame, str_column: List) -> pd.DataFrame:
    df = df_ori.copy()

    for column in str_column:
        df[column] = dfNormalize_sigmoid_module(df, column)[column]

    return df


def dfNormalize_sigmoid_module(df: pd.DataFrame, column) -> pd.DataFrame:
    # Define the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    df_copy = min_max_normalize(df, column)

    # Create a copy of the original DataFrame to keep it unaltered
    # Apply the sigmoid function to the specified column

    df_copy[column] = df_copy[column].apply(sigmoid)

    return df_copy


def normalize_sigmoid(
    df: pd.DataFrame,
    column,
    new_min=cfgs.entropy_normal_min,
    new_max=cfgs.entropy_normal_max,
    gain=1.0,
    reverse_Flag=False,
) -> pd.DataFrame:
    df_copy = min_max_normalize(df, column, def_val=1.0)

    # if cfgs.BEFORE_REL != cfgs.CURRENT_REL:
    #     cfgs.BEFORE_REL = cfgs.CURRENT_REL

    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     def sigmoid(x):
    #         return 1 / (1 + np.exp(-x))

    #     print(df)
    #     print((gain * df_copy[column]).apply(expit))

    #     x = np.arange(-6.0, 6.0, 0.1)
    #     y = sigmoid(x)
    #     plt.plot(x, y)

    #     plt.scatter(gain * df_copy[column], (gain * df_copy[column]).apply(expit), color="red")

    #     plt.scatter(gain * df_copy[column], min_max_normalize(df, column, 0, 1, def_val=1.0)[column], color="green")

    #     plt.ylim(-0.1, 1.1)

    #     plt.show()

    if reverse_Flag:
        df_copy[column] = new_max - (gain * df_copy[column]).apply(expit) * (
            new_max - new_min
        )
    else:
        df_copy[column] = (gain * df_copy[column]).apply(expit) * (
            new_max - new_min
        ) + new_min

    return df_copy


def normalize_logistic(
    df: pd.DataFrame,
    column,
    new_min=cfgs.entropy_normal_min,
    new_max=cfgs.entropy_normal_max,
    gain=1.0,
    reverse_Flag=False,
) -> pd.DataFrame:
    df_copy = min_max_normalize(df, column, 1e-2, 1 - 1e-2, def_val=1.0)
    df_copy[column] = (gain * df_copy[column]).apply(logit)
    # df_copy = min_max_normalize(df, column, 1e-3, 1 - 1e-3, def_val=1.0)

    # if cfgs.BEFORE_REL != cfgs.CURRENT_REL:
    #     cfgs.BEFORE_REL = cfgs.CURRENT_REL
    #     xdatas = min_max_normalize(df, column, 1e-2, 1 - 1e-2, def_val=1.0)
    #     plot_df = min_max_normalize(df, column, 1e-2, 1 - 1e-2, def_val=1.0)

    #     plot_df[column] = (gain * plot_df[column]).apply(logit)

    #     ydatas = min_max_normalize(plot_df, column, 0, 1, def_val=1.0, reverse_Flag=reverse_Flag)

    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     # def inv_sigmoid(x):
    #     #     return np.log(x / (1 - x))

    #     x = np.arange(0, 1.0, 0.01)
    #     y = logit(x / gain)
    #     plt.plot(x, (y + 6) / 12, color="red")

    #     xs = np.arange(-6.0, 6.0, 0.1)
    #     ys = expit(xs)
    #     plt.plot((xs + 6) / 12, ys, color="blue")

    #     df_sig = min_max_normalize(df, column, def_val=1.0)

    #     plt.scatter(xdatas[column], ydatas[column], color="red", s=50)
    #     plt.scatter(xdatas[column], (gain * df_sig[column]).apply(expit), color="blue", s=35)
    #     plt.scatter(xdatas[column], min_max_normalize(df, column, 0, 1, def_val=1.0)[column], color="green", s=15)

    #     # plt.ylim(-0.1, 1.1)

    #     plt.show()

    # df_copy[column] = (gain * df_copy[column]).apply(logit)

    # print(df_copy[column])

    df_copy = min_max_normalize(
        df_copy,
        column,
        cfgs.entropy_normal_min,
        cfgs.entropy_normal_max,
        def_val=1.0,
        reverse_Flag=reverse_Flag,
    )

    # print(df_copy[column])

    # input()

    return df_copy

    # Define the sigmoid function


# def sigmoid(x, gain=1.0):
#     return 1 / (1 + np.exp(-gain * x))


# def sigmoid_reverse(x):
#     return sigmoid(x, -1.0)


# def normalize_sigmoid_reverse(df: pd.DataFrame, column, new_min=-6, new_max=6) -> pd.DataFrame:
#     df_copy = min_max_normalize(df, column)

#     # Create a copy of the original DataFrame to keep it unaltered
#     # Apply the sigmoid function to the specified column

#     df_copy[column] = df_copy[column].apply(sigmoid_reverse) * (new_max - new_min) + new_min

#     return df_copy


def min_max_normalize(
    df_ori, column_name, new_min=-5, new_max=5, def_val=1.1, reverse_Flag=False
):
    old_min = df_ori[column_name].min()
    old_max = df_ori[column_name].max()

    df = df_ori.copy()

    if reverse_Flag:
        df[column_name] = (
            new_max
            - (df[column_name] - old_min) / (old_max - old_min) * (new_max - new_min)
        ).replace(np.nan, new_max * def_val)
    else:
        df[column_name] = (
            (df[column_name] - old_min) / (old_max - old_min) * (new_max - new_min)
            + new_min
        ).replace(np.nan, new_max * def_val)

    return df


normalize = {
    "MINMAX": min_max_normalize,
    "TB_MINMAX": min_max_normalize,
    "SIGMOID": normalize_sigmoid,
    "TB_SIGMOID": normalize_sigmoid,
    "LOGIT": normalize_logistic,
}


def min_max_normalize_reverse(df_ori, column_name, new_min=-6, new_max=6, def_val=1.1):
    old_min = df_ori[column_name].min()
    old_max = df_ori[column_name].max()

    df = df_ori.copy()

    df[column_name] = (
        new_max
        - (df[column_name] - old_min) / (old_max - old_min) * (new_max - new_min)
    ).replace(np.nan, new_max * def_val)

    return df


def to_csv_Entropy(result, path, filename):
    makedirs(f"{path}")

    if not os.path.isfile(f"{path}/{filename}"):
        first = True
    else:
        first = False

    result.to_csv(
        f"{path}/{filename}",
        mode="a",
        index=False,
        header=first,
    )


def get_row_relation(relation, strModels, filepath):
    result = pd.DataFrame(columns=cfgs.column_List_eval)

    for model in strModels:
        cfgs.entropy_df, value_dict = get_df_from_csv(
            cfgs.entropy_df, filepath, relation, model
        )

        result = pd.concat([result, value_dict], ignore_index=True)

    return result


def get_df_from_csv(entropy_csv, filename, relation_value, model="TransD"):
    if entropy_csv is None:
        df = pd.read_csv(filename)

    else:
        df = entropy_csv

    df_filtered = df[(df["relation"] == relation_value) & (df["model"] == model)]

    return df, df_filtered


def get_csv_path():
    result_path = []

    for k in cfgs.hit_k_limits:
        for str_idx in cfgs.default_entropy_strs:
            result_path += [cfgs.default_entropy_dir_path + str(k) + str_idx]

    print(result_path)

    return result_path


def get_csv_path_short():
    result_path = []

    for k in cfgs.hit_k_limits:
        for str_idx in cfgs.default_entropy_base_strs:
            result_path += [cfgs.default_entropy_dir_path + str(k) + str_idx]

    return result_path


# def get_csv_path_short(dir_path):
#     cfgs.default_entropy_dir_path = f"./csv/{dir_path}/entropy_k_"

#     result_path = []

#     for k in cfgs.hit_k_limits:
#         for str_idx in cfgs.default_entropy_base_strs:
#             result_path += [cfgs.default_entropy_dir_path + str(k) + str_idx]

#     return result_path


def check_CUDA_MEM(tag=""):
    current_device = torch.cuda.current_device()
    print(f"Line {tag}: ", torch.cuda.memory_allocated(current_device) / (1024**2))


def check_CUDA_MEM_value(value, tag=""):
    print(f"Line {tag}: ", value.element_size() * value.nelement() / (1024**2))


def tabulate_print(df):
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="psql",
        )
    )


def show_histogram(tensor: torch.Tensor, bins="auto"):
    """
    Displays a histogram for the given tensor.

    Args:
    - tensor (torch.Tensor): A tensor of size [1, n] on a "cuda:0" device.
    """

    if bins == "auto":
        # Compute the Interquartile Range (IQR)
        q75, q25 = torch.quantile(tensor, 0.75), torch.quantile(tensor, 0.25)
        iqr = q75 - q25

        # Compute bin width using Freedman-Diaconis rule
        bin_width = 2.0 * iqr * (tensor.shape[0] ** (-1 / 3))

        # Compute number of bins
        data_range = float(tensor.max() - tensor.min())

        # print(cfgs.entropy_batch)
        # print(cfgs.bins[cfgs.entropy_batch])

        if cfgs.bins[cfgs.entropy_batch] is None:
            bins = int(int(data_range / bin_width) * cfgs.bin_weight) or 1

            cfgs.bins[cfgs.entropy_batch] = bins

        else:
            bins = cfgs.bins[cfgs.entropy_batch]

    if tensor.is_cuda:
        tensor_np = tensor.detach().cpu().numpy().flatten()

    else:
        # Move tensor to CPU and convert to numpy
        tensor_np = tensor.numpy().flatten()

    if bins < 5:
        bins = int(tensor_np.shape[0] / 3) + 1

    if tensor_np.shape[0] > 50:
        bins = max(bins, 50)

    del tensor
    torch.cuda.empty_cache()

    # Plot histogram
    plt.hist(tensor_np, bins=bins, color="blue", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Tensor Values: {tensor_np.shape[0]} - {bins}")
    plt.grid(True)

    if "SHOW" in cfgs.MODE:
        plt.show()

    plt.savefig(
        f"./img_hist/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_PAIRD}_{cfgs.CURRENT_LABEL}.png",
        format="png",
    )
    plt.close()


def gaussian_pdf_using_scipy(tensor, fitter=stats.norm):
    """Calculate the Gaussian PDF of the tensor using scipy.stats"""
    # Calculate mean and standard deviation of the tensor
    # mu = torch.mean(tensor).item()  # Convert to a regular Python scalar
    # sigma = torch.std(tensor).item()  # Convert to a regular Python scalar

    # tensor_np = tensor.numpy()

    if tensor.is_cuda:
        tensor = tensor.detach().cpu()

    # tensor = torch.softmax(tensor, dim=0)

    # mu, sigma = fitter.fit(tensor)

    # Calculate the Gaussian PDF
    pdf_values = fitter.pdf(tensor.numpy(), *fitter.fit(tensor))

    del tensor
    torch.cuda.empty_cache()

    return torch.tensor(pdf_values)


def sort_by_basis(basis, data):
    # Get the indices that would sort the data array
    sorted_indices = np.argsort(basis)

    # Use the sorted indices to reorder both basis and data arrays
    sorted_basis = basis[sorted_indices]
    sorted_data = data[sorted_indices]

    return sorted_basis, sorted_data


def plots(tensor, probs):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        x=tensor.numpy(),
        y=probs * tensor.shape[0],
        color="red",
        marker="x",
        label="Data Points",
    )
    plt.autoscale()
    plt.show()
    plt.close()


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def plots_pdf_values(tensor, probs, pdf_values):
    maxs = probs.max()
    pdf_max = pdf_values.max()

    hist_values, hist_edge = np.histogram(tensor, bins="auto", density=True)

    plt.figure(figsize=(8, 6))
    # plt.scatter(tensor.numpy(), pdf_values, color="blue", label="PDF")
    # plt.hist(tensor, density=True, bins="auto", histtype="stepfilled", alpha=0.2, range=[0, maxs])

    plt.hist(
        hist_edge[:-1],
        hist_edge,
        alpha=0.6,
        color="b",
        label="Histogram of tensor",
        weights=hist_values * maxs / hist_values.max(),
    )

    plt.scatter(
        x=tensor,
        y=pdf_values * maxs / pdf_max,
        color="red",
        marker="x",
        s=10,
        label="CDF",
    )
    plt.scatter(x=tensor, y=probs, color="green", marker="o", s=10, label="Probs")
    plt.ylim([maxs * -0.2, maxs * 1.1])

    plt.title(
        f"Fitted Gaussian CDF {cfgs.CURRENT_REL} {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor.size}"
    )
    plt.xlabel("Value")
    plt.legend()

    if "SHOW" in cfgs.MODE:
        plt.show()

    if "SAVE" in cfgs.MODE:
        plt.savefig(
            f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
            format="png",
        )
    plt.close()


def getHistogram(tensor_np):
    # if "TransE" in cfgs.CURRENT_MODEL:
    #     hist_values, bin_edges = np.histogram(tensor_np, bins="auto", density=True)
    #     h_size = hist_values.size

    #     if "Euclidean" in cfgs.CURRENT_LABEL:
    #         cfgs.n_bins[cfgs.CURRENT_BATCH] = hist_values.size

    #     elif "cosine" in cfgs.CURRENT_LABEL:
    #         cfgs.c_bins[cfgs.CURRENT_BATCH] = hist_values.size

    #     else:
    #         cfgs.m_bins[cfgs.CURRENT_BATCH] = hist_values.size

    # else:
    #     if "Euclidean" in cfgs.CURRENT_LABEL:
    #         h_size = cfgs.n_bins[cfgs.CURRENT_BATCH]

    #     elif "cosine" in cfgs.CURRENT_LABEL:
    #         h_size = cfgs.c_bins[cfgs.CURRENT_BATCH]

    #     else:
    #         h_size = cfgs.m_bins[cfgs.CURRENT_BATCH]

    #     hist_values, bin_edges = np.histogram(tensor_np, bins=h_size, density=True)
    # hist_values, bin_edges = np.histogram(tensor_np, bins="auto", density=True)
    # t1, _ = np.histogram(tensor_np, bins="auto", density=True)

    # if t1.size != h_size:
    #     print(f"Diff: {cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}")
    #     print(tensor_np.size)
    #     print(f"TransE: {h_size}\n {cfgs.CURRENT_MODEL}: {t1.size}")

    # return hist_values, bin_edges

    return np.histogram(tensor_np, bins="auto", density=True)


def get_hist_data(tensor_np, hist_values, bin_edges):
    bin_indices = np.digitize(tensor_np, bin_edges) - 1

    bin_indices[bin_indices == len(hist_values)] = len(hist_values) - 1

    bin_indices[bin_indices < 0] = 0

    # Map bin indices to hist values to get the histogram value for each item in tensor_np
    hist_values_for_each_item = hist_values[bin_indices]

    return hist_values_for_each_item


def plot_gaussian(_tensor: torch.tensor, pdf_values, fitter=stats.norm):
    if _tensor.is_cuda:
        tensor = _tensor.detach().cpu()
    else:
        tensor = _tensor

    # if _pdf_values.is_cuda:
    #     pdf_values = _pdf_values.detach().cpu()

    # else:
    #     pdf_values = _pdf_values

    # Generate X values for plotting
    x = np.linspace(
        tensor.min().item() - 3 * tensor.std().item(),
        tensor.max().item() + 3 * tensor.std().item(),
        1000,
    )

    # print(fitter.fit(tensor))
    # mu, sigma = fitter.fit(tensor)

    # Compute corresponding Y values for the Gaussian curve
    y = fitter.pdf(x, *fitter.fit(tensor))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="PDF")
    plt.hist(tensor, density=True, bins="auto", histtype="stepfilled", alpha=0.2)
    plt.scatter(
        tensor.numpy(), pdf_values, color="red", marker="o", label="Data Points"
    )
    plt.title(
        f"PDF: {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor.shape[0]} {cfgs.CURRENT_LABEL}"
    )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    if "SHOW" in cfgs.MODE:
        plt.show()

    if "SAVE" in cfgs.MODE:
        plt.savefig(
            f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
            format="png",
        )

        # plt.savefig(
        #     f"./img_hist/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_PAIRD}_{cfgs.CURRENT_LABEL}.jpg",
        #     format="jpeg",
        # )

    del _tensor

    torch.cuda.empty_cache()

    plt.close()


def dataLoader(strDataset):
    if "FB" in strDataset:
        return TrainDataLoader(
            in_path=f"./benchmarks/{strDataset}/",
            nbatches=100,
            threads=8,
            sampling_mode="normal",
            bern_flag=1,
            filter_flag=1,
            neg_ent=25,
            neg_rel=0,
        )

    else:
        return TrainDataLoader(
            in_path=f"./benchmarks/WN18RR/",
            nbatches=256,
            threads=8,
            sampling_mode="normal",
            bern_flag=1,
            filter_flag=1,
        )


def comp_plots_pdf_values(tensor, probs, fig, ax1):
    print(type(ax1))

    if cfgs.CURRENT_LABEL == "Euclidean":
        colors = "blue"
        color2 = "b"

    else:
        colors = "r"
        color2 = "y"

    maxs = probs.max().item()

    hist_values, hist_edge = np.histogram(tensor, bins="auto", density=True)

    # plt.figure(figsize=(8, 6))

    ax1.hist(
        hist_edge[:-1],
        hist_edge,
        alpha=0.4,
        color=color2,
        label=f"Histogram of {cfgs.CURRENT_LABEL}",
        weights=hist_values * maxs / hist_values.max(),
    )

    ax1.scatter(
        x=tensor,
        y=probs,
        color=colors,
        marker="o",
        s=2,
        label=f"Probability of {cfgs.CURRENT_LABEL}",
    )
    ax1.set_ylim([maxs * -0.05, maxs * 1.1])

    # return ax1

    # plt.show()

    # print("File name:")
    # print(
    #     f"./img_hist/{cfgs.dataset}/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/Resize/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
    # )

    # return plt


# def comp_plots(tensor_Euclidean, probs_Euclidean, tensor_cos, probs_cos):

#     comp_plots_pdf_values(tensor_Euclidean, probs_Euclidean, colors="blue", labels = "Euclidean", limit_img=0)


if __name__ == "__main__":
    # dataset_name = "FB15K237"
    # load_dataset_rel(dataset_name, "train2id", "22")

    # entropy_df = {}
    # for path in get_csv_path():
    #     entropy_df[path] = pd.read_csv(path)

    print(get_csv_path_short())
