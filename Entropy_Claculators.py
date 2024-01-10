import concurrent.futures
import itertools
import json
import math
import os
import pathlib
import pickle
import random
import time
from itertools import islice
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.cuda as cuda
import torch.cuda.memory as memory
import torch.distributions as distributions
import torch.distributions as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

import config_TransEE as cfgs
import utilities as util
from api_entropies import EntropyCalculator

# dataset = "FB15K237"
column_List = [
    "model",
    "relation",
    "num_correct",
    "n_shannon",
    "c_shannon",
    "cs_shannon",
    "n_condi",
    "c_condi",
    "cs_condi",
    "n_joint",
    "c_joint",
    "cs_joint",
    "n_renyi",
    "c_renyi",
    "cs_renyi",
    "n_tsallis",
    "c_tsallis",
    "cs_tsallis",
    "n_diff",
    "c_diff",
    "cs_diff",
]

entity_tot = 0
relation_tot = 0

# entropyCalculator = EntropyCalculator()

dataFile = "train2id"


def load_data(i):
    global dataFile
    head_entities, tail_entities = util.load_dataset_rel(cfgs.dataset, dataFile, str(i))

    # print(dataFile)
    # head_entities, tail_entities = util.load_dataset_rel(dataset, "valid2id", str(i))

    result = {
        "batch_h": head_entities.astype(np.int64),
        "batch_t": tail_entities.astype(np.int64),
        "batch_r": np.array([i], dtype=np.int64),
        "mode": "head_batch",
    }

    return i, result


def get_train_data(filename="relation2id"):
    triple = open(f"./benchmarks/{cfgs.dataset}/{filename}.txt", "r")
    tot = (int)(triple.readline())
    triple.close()

    datas = {}
    # data_tmp = {}
    global dataFile

    dataFile = "train2id"
    count = 0

    with Pool(int(cpu_count() / 2)) as pool:
        # count = 0
        for i, result in pool.imap_unordered(load_data, range(tot)):
            datas[i] = result
            # data_tmp[i] = result
            # count += len(result["batch_h"])

    # print(len(datas[0]["batch_t"]))

    if "MIXED" in cfgs.MODE_DATA:
        dataFile = "valid2id"
        with Pool(int(cpu_count() / 2)) as pool:
            for j, result in pool.imap_unordered(load_data, range(tot)):
                datas[j]["batch_h"] = np.append(datas[j]["batch_h"], result["batch_h"])
                datas[j]["batch_t"] = np.append(datas[j]["batch_t"], result["batch_t"])

                # datas[j + i] = result
                count += len(result["batch_h"])

    # print(len(datas[0]["batch_t"]))

    return datas, count


def isSucessfulPredictions(top_k, result, options=True):
    strResult = str((top_k["tail_id"] == result).any())
    if strResult != "False":
        bool_series = top_k["tail_id"] == result
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


def _prepare_hrt_for_functional(
    resources, entity_id, relation_id, entity_tot, mode="tail_batch"
):
    if mode == "tail_batch":
        prediction_data = {
            "batch_h": util.to_var(
                np.array([entity_id], dtype=np.int64),
                True,
            ),
            "batch_r": util.to_var(resources[relation_id]["batch_r"], True),
            "batch_t": util.to_var(np.array(range(0, entity_tot)), True),
            "mode": "tail_batch",
        }
    else:
        prediction_data = {
            "batch_h": util.to_var(np.array(range(0, entity_tot)), True),
            "batch_r": util.to_var(resources[relation_id]["batch_r"], True),
            "batch_t": util.to_var(
                np.array([entity_id], dtype=np.int64),
                True,
            ),
            "mode": "head_batch",
        }

    return prediction_data


def _prepare_hrt_for_functional_direct_single(entity, relation, entity_tot, mode):
    rels = util.to_var_single([relation], True)

    if mode == "tail_batch":
        ent_str = "batch_h"
        tar_str = "batch_t"

    else:
        ent_str = "batch_t"
        tar_str = "batch_h"

    if str(entity.device) == "cpu":
        entity = entity.to("cuda:0")

    if entity.shape[0] != 0:
        prediction = {
            ent_str: entity,
            "batch_r": rels,
            tar_str: util.to_var(np.array(range(0, entity_tot)), True),
            "mode": mode,
        }

    del entity
    del rels
    torch.cuda.empty_cache()

    return prediction


def _predict_pretrained_model(
    model, entity_id, relation_id, resources, entity_tot, mode="tail_batch"
):
    prediction_data = _prepare_hrt_for_functional(
        resources, entity_id, relation_id, entity_tot, mode
    )

    prediction = model.predict(prediction_data)

    predicted_df = (
        pd.DataFrame(prediction).sort_values(by=[0], ascending=True).reset_index()
    )

    predicted_df.rename(columns={"index": "tail_id"}, inplace=True)
    predicted_df.rename(columns={0: "score"}, inplace=True)

    return predicted_df


def predict_pretrained_model(
    model_pre,
    train_dataloader,
    trained_datas,
    mode="tail_batch",
    stopper=cfgs.rel_stopper_index,
):
    global relation_tot
    global entity_tot

    entity_tot = train_dataloader.get_ent_tot()
    relation_tot = train_dataloader.get_rel_tot()
    triple_tot = train_dataloader.get_triple_tot()

    results = {}

    print("stopper: ", stopper)

    if mode == "tail_batch":
        batch = "batch_h"
        batch_traget = "batch_t"

    else:
        batch = "batch_t"
        batch_traget = "batch_h"

    with tqdm(total=triple_tot) as pbar:
        for rel_index in range(0, relation_tot):
            if rel_index >= stopper:
                break

            results[rel_index] = pd.DataFrame(
                columns=["entity", "relation", "Hitk", "mode"]
            )
            for i, entity_index in enumerate(trained_datas[rel_index][batch]):
                df_result = _predict_pretrained_model(
                    model_pre,
                    entity_index,
                    rel_index,
                    trained_datas,
                    entity_tot,
                    mode=mode,
                )

                result = pd.DataFrame(
                    {
                        "entity": [entity_index],
                        "relation": [rel_index],
                        "Hitk": [
                            util.isSucessfulPredictions(
                                df_result,
                                trained_datas[rel_index][batch_traget][i],
                                False,
                            )
                        ],
                        "mode": [mode],
                    }
                )

                results[rel_index] = pd.concat(
                    [results[rel_index], result], ignore_index=True
                )

                pbar.update(1)

            results[rel_index] = (
                results[rel_index]
                .sort_values(by=["Hitk"], ascending=True)
                .reset_index(drop=True)
            )

    return results


def predict_pretrained_model_mixed(
    model_pre,
    train_dataloader,
    trained_datas,
    stopper=cfgs.rel_stopper_index,
    ad_count=0,
):
    global relation_tot
    global entity_tot

    triple_tot = train_dataloader.get_triple_tot() + ad_count

    results = {}

    print("tot: ", train_dataloader.get_triple_tot())

    print("ad_count: ", ad_count)
    print("triple_tot: ", triple_tot)
    print("relation_tot: ", relation_tot)
    print("stopper: ", stopper)

    # input()

    with tqdm(total=triple_tot * 2) as pbar:
        for rel_index in range(0, relation_tot):
            pbar.set_description(str(rel_index))

            if cfgs.hold_index is not None:
                rel_index = cfgs.hold_index

            if rel_index >= stopper:
                break

            results[rel_index] = pd.DataFrame(
                columns=["entity", "relation", "Hitk", "mode"]
            )
            for strMode in ["tail_batch", "head_batch"]:
                if strMode == "tail_batch":
                    batch = "batch_h"
                    batch_traget = "batch_t"

                else:
                    batch = "batch_t"
                    batch_traget = "batch_h"

                for i, entity_index in enumerate(trained_datas[rel_index][batch]):
                    df_result = _predict_pretrained_model(
                        model_pre,
                        entity_index,
                        rel_index,
                        trained_datas,
                        entity_tot,
                        mode=strMode,
                    )

                    result = pd.DataFrame(
                        {
                            "entity": [entity_index],
                            "relation": [rel_index],
                            "Hitk": [
                                util.isSucessfulPredictions(
                                    df_result,
                                    trained_datas[rel_index][batch_traget][i],
                                    False,
                                )
                            ],
                            "mode": [strMode],
                        }
                    )

                    results[rel_index] = pd.concat(
                        [results[rel_index], result], ignore_index=True
                    )

                    pbar.update(1)

                # print(results[rel_index])
                # input()

            if cfgs.debug_flag:
                break

            results[rel_index] = (
                results[rel_index]
                .sort_values(by=["Hitk"], ascending=True)
                .reset_index(drop=True)
            )

            # pd.set_option("display.max_rows", None)
            # print(results[rel_index])
            # input()

    return results


def pretrained_results(stopper=cfgs.rel_stopper_index):
    global relation_tot
    global entity_tot

    # util.endl("initializing")
    trained_datas, ad_count = get_train_data()

    # util.endl(f"add_count {ad_count}")

    models = {}

    result = {}

    for strModel in cfgs.strModels:
        model, train_dataloader = util.load_models[cfgs.dataset][strModel](
            cfgs.data_tag
        )

        models[strModel] = model

        entity_tot = train_dataloader.get_ent_tot()
        relation_tot = train_dataloader.get_rel_tot()

        if cfgs.Mode_Calculator == "Read":
            with open(
                f"./resource_pkl/{cfgs.dataset}/{cfgs.MODE_DATA}/result.pkl", "rb"
            ) as f:
                result = pickle.load(f)

        else:
            result[strModel] = predict_pretrained_model_mixed(
                model,
                train_dataloader,
                trained_datas,
                stopper=stopper,
                ad_count=ad_count,
            )

        del model
        torch.cuda.empty_cache()

    if cfgs.Mode_Calculator == "Write":
        util.makedirs(f"./resource_pkl/{cfgs.dataset}/{cfgs.MODE_DATA}")
        with open(
            f"./resource_pkl/{cfgs.dataset}/{cfgs.MODE_DATA}/result.pkl", "wb"
        ) as f:
            pickle.dump(result, f)

    return models, result


def load_and_predict_model(args):
    strModel, trained_datas, stopper = args
    global relation_tot
    global entity_tot

    model, train_dataloader = util.load_model[strModel](cfgs.data_tag)

    entity_tot = train_dataloader.get_ent_tot()
    relation_tot = train_dataloader.get_rel_tot()

    if cfgs.Mode_Calculator == "Read":
        with open(
            f"./resource_pkl/{cfgs.dataset}/{cfgs.MODE_DATA}/result.pkl", "rb"
        ) as f:
            result = pickle.load(f)
    else:
        result = predict_pretrained_model_mixed(
            model, train_dataloader, trained_datas, stopper=stopper
        )

    torch.cuda.empty_cache()

    return strModel, model, result


def pretrained_results_pool(stopper=400):
    trained_datas, ad_count = get_train_data()

    # Prepare arguments as tuples
    args = [(strModel, trained_datas, stopper) for strModel in cfgs.strModels]

    with Pool(processes=len(cfgs.strModels)) as pool:
        results = pool.map(load_and_predict_model, args)

    models = {}
    result = {}

    for r in results:
        strModel, model, model_result = r
        models[strModel] = model
        if cfgs.Mode_Calculator != "Read":
            result[strModel] = model_result

    if cfgs.Mode_Calculator == "Write":
        with open(f"./resource_pkl/{cfgs.MODE_DATA}/result.pkl", "wb") as f:
            pickle.dump(result, f)

    return models, result


def entropy_calc(
    resource_tensor,
    strModel,
    relation,
    minSize,
    correctSize,
    entropy_names=cfgs.str_entropy_output_default,
    hit_limit=10,
):
    # Initialize result DataFrame with common columns

    result_columns = {
        "model": strModel,
        "relation": relation,
        "num_correct": minSize,
        "num_ori": correctSize,
    }

    if minSize >= 5:
        entropyCalculator = EntropyCalculator()

        # print("correctSize: ", correctSize)
        entropyCalculator.set_resource(resource_tensor, minSize)

        # Unpacking calculated entropies
        entropies = entropyCalculator.calculate_entropie[cfgs.CALC_MODE]()

        # Adding entropies to result columns
        result_columns.update(
            {name: [value] for name, value in zip(entropy_names, entropies)}
        )

        # Cleaning up
        del entropyCalculator
        del resource_tensor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        # Assigning 0.0 to entropies in case of minSize < 10
        zero_entropies = {name: [0.0] for name in entropy_names}
        result_columns.update(zero_entropies)

    result = pd.DataFrame(result_columns)

    return result


def div_Target_df(df, tail_min_length, head_min_length):
    tail_batch = df[df["mode"] == "tail_batch"]
    head_batch = df[df["mode"] == "head_batch"]

    tail_batch = tail_batch[:tail_min_length]
    head_batch = head_batch[:head_min_length]

    tail_batch = util.to_var(tail_batch["entity"].values.reshape(-1, 1), False)
    head_batch = util.to_var(head_batch["entity"].values.reshape(-1, 1), False)

    return tail_batch, head_batch


def div_Target_df_single(df, mode, limit, mins):
    # print(df)

    batch = df[(df["mode"] == mode) & (df["limit"] == limit)].reset_index(drop=True)

    ori_size = len(batch)

    batch = util.to_var(batch["entity"].values.reshape(-1, 1), False)

    return batch, ori_size


def to_csv_Entropy(result, hit, rel, strModel, tag):
    if rel == 0 and cfgs.strModels.index(strModel) == 0:
        first = True
    else:
        first = False

    path = f"./csv/{cfgs.dataset}/{cfgs.date}_{cfgs.alpha}_{cfgs.q}"

    util.makedirs(path)

    result.to_csv(
        f"{path}/entropy_k_" + str(hit) + str(tag) + ".csv",
        mode="a",
        index=False,
        header=first,
    )


def save_entropy(
    resource,
    strModel,
    rel,
    min_length,
    correctSize,
    hit,
    strBatch,
    rank="None",
    paired=False,
):
    # correctSize = prediction_lists[hit][str_ent].shape
    # util.check_CUDA_MEM("save_entropy 1")

    # print(cfgs.MODE)

    result = entropy_calc(
        resource,
        strModel,
        rel,
        min_length,
        correctSize,
        cfgs.entropy_output[cfgs.CALC_MODE],
        hit_limit=hit,
    )
    # util.check_CUDA_MEM("save_entropy")

    # util.endl(f"{rel} / {cfgs.CURRENT_BATCH} - save_entropy")

    if "PRINT_ENTROPY" in cfgs.MODE:
        print(result)

    if "STAY" in cfgs.MODE:
        input()

    if paired:
        path = f"_{strBatch}_paired"
    else:
        path = f"_{strBatch}"

    # input()

    if "EMPTY" not in cfgs.MODE:
        to_csv_Entropy(result, hit, rel, strModel, path)

    del resource
    torch.cuda.empty_cache()
    # print("Save Done")


def get_entropy_model(
    models, pretrained_hit_ks, hit_k_limits=[100000], stopper=cfgs.rel_stopper_index
):
    global entity_tot
    global relation_tot

    relation_length = int(
        (relation_tot - cfgs.Start_Index) * len(cfgs.strModels) * len(hit_k_limits) * 2
    )

    print("Length:\t", len(cfgs.strModels) * len(hit_k_limits) * 2)
    print("Total:\t", relation_length)

    with tqdm(total=relation_length) as pbar:
        for rel in range(cfgs.Start_Index, relation_tot):
            min_hits = {}

            min_hits["tail_batch"] = {}
            min_hits["head_batch"] = {}

            # input()
            if cfgs.hold_index is not None:
                rel = cfgs.hold_index

            if rel >= stopper:
                break

            cfgs.CURRENT_REL = rel

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            hits_lists = {}

            for strModel in cfgs.strModels:
                # pbar.set_description(f"{strModel} - {rel}")

                hits_list = pd.DataFrame(columns=["limit", "entity", "rank", "mode"])

                count = 0

                model_hits = []

                if cfgs.NEW_AVG == "Total":
                    hits = 0

                    for idx, val in pretrained_hit_ks[strModel][rel].iterrows():
                        count += 1

                        for hit in hit_k_limits:
                            model_hits.append(
                                [hit, val["entity"], val["Hitk"], val["mode"]]
                            )

                    hits_df = pd.DataFrame(
                        model_hits, columns=["limit", "entity", "rank", "mode"]
                    )
                    hits_df = (
                        hits_df.groupby(["limit", "entity", "mode"])["rank"]
                        .mean()
                        .reset_index()
                        .sort_values(by=["rank"], ascending=True)
                        .reset_index(drop=True)
                    )

                    print(len(hits_df))

                    # hits_lists[strModel]

                    for strBatch in ["tail_batch", "head_batch"]:
                        hits = hit

                        i_hits_df = hits_df[hits_df["mode"] == strBatch]
                        pd.set_option("display.max_rows", None)

                        while True:
                            if (
                                len(i_hits_df[i_hits_df["rank"] <= hits]) < 2000
                                or hits == 1
                            ):
                                break

                            else:
                                hits -= 1

                        while True:
                            if (
                                len(i_hits_df[i_hits_df["rank"] <= hits]) >= 10
                                or hits >= 20 * hit
                            ):
                                break

                            else:
                                hits += 1

                        i_hits_df = i_hits_df[i_hits_df["rank"] <= hits]

                        if "tail" in strBatch:
                            hits_lists[strModel] = i_hits_df

                        else:
                            hits_lists[strModel] = pd.concat(
                                [hits_lists[strModel], i_hits_df], ignore_index=True
                            )

                elif cfgs.NEW_AVG == "TotPer":
                    hits = 0

                    for idx, val in pretrained_hit_ks[strModel][rel].iterrows():
                        count += 1

                        for hit in hit_k_limits:
                            model_hits.append(
                                [hit, val["entity"], val["Hitk"], val["mode"]]
                            )

                    hits_df = pd.DataFrame(
                        model_hits, columns=["limit", "entity", "rank", "mode"]
                    )
                    hits_df = (
                        hits_df.groupby(["limit", "entity", "mode"])["rank"]
                        .mean()
                        .reset_index()
                        .sort_values(by=["rank"], ascending=True)
                        .reset_index(drop=True)
                    )

                    # print(len(hits_df))

                    # hits_lists[strModel]

                    for strBatch in ["tail_batch", "head_batch"]:
                        hits = hit

                        i_hits_df = hits_df[hits_df["mode"] == strBatch].reset_index(
                            drop=True
                        )

                        lims = min(
                            max(
                                len(i_hits_df[i_hits_df["rank"] < i_hits_df["limit"]])
                                / len(i_hits_df),
                                1 / len(i_hits_df),
                            ),
                            1.0,
                        )

                        # print(i_hits_df)

                        i_hits_df, a = get_smallest_percentile(
                            i_hits_df, "rank", percentile=lims
                        )

                        i_hits_df = i_hits_df[
                            i_hits_df["rank"] <= cfgs.NWeight * i_hits_df["limit"]
                        ]

                        min_hits[strBatch][strModel] = i_hits_df["rank"].max()
                        # print(i_hits_df)

                        if "tail" in strBatch:
                            hits_lists[strModel] = i_hits_df

                        else:
                            hits_lists[strModel] = pd.concat(
                                [hits_lists[strModel], i_hits_df], ignore_index=True
                            )
                        # print(i_hits_df, min_hits[strBatch][strModel])
                        # print(strBatch, hits, "-", len(i_hits_df))
                        # input()

                elif cfgs.NEW_AVG == "Pers":
                    hits = 0

                    for idx, val in pretrained_hit_ks[strModel][rel].iterrows():
                        count += 1

                        for hit in hit_k_limits:
                            model_hits.append(
                                [hit, val["entity"], val["Hitk"], val["mode"]]
                            )

                    hits_df = pd.DataFrame(
                        model_hits, columns=["limit", "entity", "rank", "mode"]
                    )
                    hits_df = (
                        hits_df.groupby(["limit", "entity", "mode"])["rank"]
                        .mean()
                        .reset_index()
                        .sort_values(by=["rank"], ascending=True)
                        .reset_index(drop=True)
                    )

                    # print(len(hits_df))

                    # hits_lists[strModel]

                    for strBatch in ["tail_batch", "head_batch"]:
                        hits = hit

                        i_hits_df = hits_df[hits_df["mode"] == strBatch].reset_index(
                            drop=True
                        )

                        # print(i_hits_df)
                        # pd.set_option("display.max_rows", None)
                        # print(i_hits_df)
                        # input()

                        i_hits_df, _ = get_smallest_percentile(i_hits_df, "rank")

                        if "tail" in strBatch:
                            hits_lists[strModel] = i_hits_df

                        else:
                            hits_lists[strModel] = pd.concat(
                                [hits_lists[strModel], i_hits_df], ignore_index=True
                            )

                elif cfgs.NEW_AVG == "Half":
                    for idx, val in pretrained_hit_ks[strModel][rel].iterrows():
                        count += 1

                        for hit in hit_k_limits:
                            if int(val["Hitk"]) < 2 * hit:
                                new_row = pd.DataFrame(
                                    [[hit, val["entity"], val["Hitk"], val["mode"]]],
                                    columns=["limit", "entity", "rank", "mode"],
                                )

                                hits_list = pd.concat(
                                    [hits_list, new_row], ignore_index=True
                                )

                    hits_df = (
                        hits_list.groupby(["limit", "entity", "mode"])["rank"]
                        .mean()
                        .reset_index()
                        .sort_values(by=["rank"], ascending=True)
                        .reset_index(drop=True)
                    )

                    for strBatch in ["tail_batch", "head_batch"]:
                        hits = hit

                        i_hits_df = hits_df[hits_df["mode"] == strBatch]

                        while True:
                            i_hits_df = i_hits_df[i_hits_df["rank"] <= hits]

                            # print("1: ", hits, "-", len(i_hits_df))

                            if len(i_hits_df) < 2000 or hits == 1:
                                break

                            else:
                                hits -= 1

                        while True:
                            i_hits_df = i_hits_df[i_hits_df["rank"] <= hits]

                            # print("2: ", hits, "-", len(i_hits_df))

                            if len(i_hits_df) >= 10 or hits >= 2 * hit:
                                break

                            else:
                                hits += 1

                        if "tail" in strBatch:
                            hits_lists[strModel] = i_hits_df

                        else:
                            hits_lists[strModel] = pd.concat(
                                [hits_lists[strModel], i_hits_df], ignore_index=True
                            )

                        # print(strBatch, hits, "-", len(i_hits_df))

                else:
                    # print("ELSE INIT")
                    for idx, val in pretrained_hit_ks[strModel][rel].iterrows():
                        count += 1

                        for hit in hit_k_limits:
                            if int(val["Hitk"]) < hit:
                                new_row = pd.DataFrame(
                                    [[hit, val["entity"], val["Hitk"], val["mode"]]],
                                    columns=["limit", "entity", "rank", "mode"],
                                )

                                hits_list = pd.concat(
                                    [hits_list, new_row], ignore_index=True
                                )

                    hits_lists[strModel] = (
                        hits_list.groupby(["limit", "entity", "mode"])["rank"]
                        .mean()
                        .reset_index()
                        .sort_values(by=["rank"], ascending=True)
                        .reset_index(drop=True)
                    )

            if cfgs.MIN_HIT_PAIRED:
                nan_counts = {}

                for strBatch in ["tail_batch", "head_batch"]:
                    nan_count = 0
                    for value in min_hits[strBatch].values():
                        if np.isnan(value):
                            nan_count += 1

                    nan_counts[strBatch] = nan_count

                #     print(min_hits[strBatch])

                # print(nan_counts)
                # input()

                hits_df = {}

                for strModel in cfgs.strModels:
                    filtered_model = pd.DataFrame(
                        columns=["limit", "entity", "mode", "rank"]
                    )

                    # print("Source", hits_lists[strModel])

                    for strBatch in ["tail_batch", "head_batch"]:
                        if nan_counts[strBatch] >= 4:
                            break

                        pmin_value = min(
                            value
                            for value in min_hits[strBatch].values()
                            if not math.isnan(value)
                        )
                        # print(
                        #     hits_lists[strModel][
                        #         hits_lists[strModel]["mode"] == strBatch
                        #     ]
                        # )
                        # print(pmin_value)

                        filtered_df = hits_lists[strModel][
                            (hits_lists[strModel]["mode"] == strBatch)
                            & (hits_lists[strModel]["rank"] <= pmin_value)
                        ]

                        filtered_model = pd.concat(
                            [filtered_model, filtered_df], ignore_index=True
                        )
                        # print("Result", filtered_df)
                        # input()

                    hits_lists[strModel] = filtered_model

                # t_min = min()

                # hits_lists[strModel] = hits_lists[strModel][hits_lists[strModel]["rank"]<hit]

                # print(hits_lists[strModel])
                # input()

                # for hit in hit_k_limits:
                #     print(hits_lists[strModel][hits_lists[strModel]["limit"] == hit])

                # input()

                # util.endl_time("Init pre done")

                # pd.set_option("display.max_rows", None)
                # print(hits_lists[strModel])
                # input()

            for strModel in cfgs.strModels:
                hits_lists[strModel] = hits_lists[strModel].astype(
                    {"limit": int, "entity": int, "mode": object, "rank": int}
                )

            for hit in hit_k_limits:
                min_length = {
                    strBatch: util.get_min_length(strBatch, hit, hits_lists)
                    for strBatch in ["tail_batch", "head_batch"]
                }
                cfgs.bins = {"tail_batch": None, "head_batch": None}
                cfgs.CURRENT_HIT = hit

                for strModel in cfgs.strModels:
                    # util.check_CUDA_MEM("2-2 Defualt")
                    cfgs.CURRENT_MODEL = strModel

                    model = models[strModel]

                    trained_resource = {
                        "tail_batch": torch.tensor([]),
                        "head_batch": torch.tensor([]),
                    }

                    mix_correctSize = 0

                    # print(strModel)

                    # util.check_CUDA_MEM("2-2-3")
                    for strBatch in ["tail_batch", "head_batch"]:
                        pbar.set_description(f"{strModel}: {strBatch} + {rel}")

                        util.endl_mode(f"{rel} {hit} {strModel} {strBatch}")
                        # util.check_CUDA_MEM("")
                        cfgs.CURRENT_BATCH = strBatch

                        cfgs.entropy_batch = strBatch

                        # print(hits_lists[strModel])
                        # print(hits_lists[strModel], strBatch, hit, min_length)
                        # input()

                        batch, correctSize = div_Target_df_single(
                            hits_lists[strModel], strBatch, hit, min_length
                        )

                        if correctSize > 0:
                            mix_correctSize += correctSize

                            # print(len(batch))

                            _batch = _prepare_hrt_for_functional_direct_single(
                                batch,
                                rel,
                                entity_tot,
                                strBatch,
                            )
                            if "PAIRED" not in cfgs.MODE:
                                cfgs.CURRENT_PAIRD = "Not_Paired"

                            else:
                                cfgs.CURRENT_PAIRD = "Paired"

                            trained_resource[strBatch] = model.predict_entropy(
                                _batch
                            ).squeeze(1)

                            save_entropy(
                                trained_resource[strBatch],
                                strModel,
                                rel,
                                min_length[strBatch],
                                correctSize,
                                hit,
                                strBatch,
                                rank="None",
                            )

                            # else:
                            #     cfgs.CURRENT_PAIRD = "Paired"

                            #     trained_resource[strBatch] = model.predict_entropy(
                            #         {
                            #             **_batch,
                            #             "batch_h": _batch["batch_h"][
                            #                 : min_length[strBatch]
                            #             ],
                            #         }
                            #     ).squeeze(1)

                            #     del _batch
                            #     torch.cuda.empty_cache()

                            #     save_entropy(
                            #         trained_resource[strBatch],
                            #         strModel,
                            #         rel,
                            #         min_length[strBatch],
                            #         correctSize,
                            #         hit,
                            #         strBatch,
                            #         rank="None",
                            #         paired=True,
                            #     )

                        # util.check_CUDA_MEM("2-2-3-6")

                        del batch
                        torch.cuda.empty_cache()
                        pbar.update(1)
                        # util.check_CUDA_MEM("2-2-3-7")

                        # if "SINGLE" in cfgs.MODE:
                        #     break

                    # mixed_resource = torch.cat(
                    #     (
                    #         trained_resource["tail_batch"],
                    #         trained_resource["head_batch"],
                    #     ),
                    #     0,
                    # )
                    # # util.check_CUDA_MEM("2-2-4")

                    # save_entropy(
                    #     mixed_resource,
                    #     strModel,
                    #     rel,
                    #     min_length["tail_batch"] + min_length["head_batch"],
                    #     mix_correctSize,
                    #     hit,
                    #     "mixed_batch",
                    #     rank="None",
                    # )

                    # util.check_CUDA_MEM("2-2-5")
                    del trained_resource["tail_batch"]
                    del trained_resource["head_batch"]
                    del trained_resource
                    # del mixed_resource

                    torch.cuda.empty_cache()

            if cfgs.debug_flag:
                break


def get_smallest_percentile(df, column, percentile=0.1):
    subset, max_value = _get_smallest_percentile(df, column, percentile)

    # print("T - ", max_value)

    # Adjust the percentage if necessary

    # count = 0
    while (
        max_value < cfgs.hit_k_limits[0] * cfgs.NWeight
        and percentile * 1.25 <= 1
        and percentile <= 0.2
    ):
        percentile *= 1.25

        threshold = df[column].quantile(percentile)
        subset = df[df[column] <= threshold]

        # if max_value == subset[column].max():
        #     count += 1
        # else:
        #     count = 0

        max_value = subset[column].max()
        subset = df[df[column] <= max_value]

    return subset, max_value


def _get_smallest_percentile(df, column, percentile=0.1, ntag=1000):
    lens = percentile * len(df)
    percentile = percentile if lens < ntag else ntag / len(df)

    # Calculate the percentile value
    threshold = df[column].quantile(percentile)

    # Filter the DataFrame
    subset = df[df[column] <= threshold]

    # Handle ties
    max_value = subset[column].max()
    subset = df[df[column] <= max_value]

    if len(subset) >= ntag * 2 and max_value > 1:
        subset, max_value = _get_smallest_percentile(
            df, column, percentile * 0.75, ntag
        )

    return subset, max_value


def run_learning(
    strDataset="FB15K237",
    strType="DROP_RES_PDF_Categorical",
    strMode="RUN",
    strCalcMode="Write",
    strDateTag="_Mixed",
    strMODE_DATA="MIXED",
    mode_Min_Resource=cfgs.MODE_MIN_RESOURCE,
):
    TYPE = strType
    # TYPE = "test"
    # cfgs.setDataset("WN18RR")
    # cfgs.setDataset("FB15K")

    cfgs.setDataset(strDataset)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfgs.CALC_MODE = "DEFAULT_NO_MAHA"
    cfgs.MODE = strMode
    # cfgs.MODE = "EMPTY_IMAGE_SHOW"
    # cfgs.MODE = "EMPTY"
    cfgs.CURRENT_ENTROPY_SELECTOR = "pdf"
    cfgs.MODE_MIN_RESOURCE = mode_Min_Resource

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfgs.date = f"{TYPE}{strDateTag}"
    cfgs.MODE_DATA = strMODE_DATA
    cfgs.Mode_Calculator = strCalcMode

    util.check_CUDA_MEM(f"Start {cfgs.CURRENT_ENTROPY_SELECTOR} {cfgs.date}")

    if cfgs.hold_index is not None:
        cfgs.debug_flag = True

    else:
        cfgs.debug_flag = False

    models, pretrained_hit_ks = pretrained_results()

    util.endl("Pretrained prediction was finished")
    get_entropy_model(models, pretrained_hit_ks, cfgs.hit_k_limits)

    util.check_CUDA_MEM("End")


def run_set(
    strDataset="FB15K237",
    types="DROP_RES_PDF_Categorical",
    calcMode="Write",
    MIN_RES=0,
    MODE=["RUN"],
):
    if "WN" in strDataset:
        cfgs.hit_k_limits = [3]

    else:
        cfgs.hit_k_limits = [5]

    for mode in MODE:
        run_learning(
            strDataset=strDataset,
            strType=types,
            strMode=mode,
            strCalcMode=calcMode,
            strDateTag="_Trained",
            strMODE_DATA="TRAINING",
            mode_Min_Resource=MIN_RES,
        )


if __name__ == "__main__":
    # cfgs.NEW_AVG = "Pers"
    # types = "PER_PDF"
    # for ds in ["FB15K237", "FB15K", "WN18RR"]:
    #     run_set(
    #         ds,
    #         types=types,
    #         calcMode="Read",
    #         MIN_RES=0,
    #         MODE=["RUN"],
    #     )

    cfgs.change_Hit_Mode("TotPer")
    # cfgs.change_Hit_Mode("None")

    cfgs.MIN_HIT_PAIRED = False
    print(cfgs.NEW_AVG)

    types = "FINAL_NP_PDF"
    for ds in ["FB15K237", "FB15K", "WN18RR"]:
        # for ds in ["WN18RR"]:
        run_set(
            ds,
            types=types,
            calcMode="Read",
            MIN_RES=0,
            # MODE=["RUN"],
            MODE=["RUN"],
        )

    # for ds in ["FB15K237", "FB15K", "WN18RR"]:
    #     run_learning(
    #         strDataset=ds,
    #         strMode="EMPTY_PAIRED_IMAGE_SAVE",
    #         strCalcMode="Read",
    #         strDateTag="_Trained",
    #         strMODE_DATA="TRAINING",
    #         mode_Min_Resource=0,
    #     )
