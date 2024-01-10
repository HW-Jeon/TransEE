import importlib

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config_TransEE as cfgs
import utilities as util
from openke.module.model import TransD, TransE, TransH, TransR

from .Model import Model


class TransEE(Model):
    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=200,
        devices="cuda:0",
    ):
        super(TransEE, self).__init__(ent_tot, rel_tot)

        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim, device=devices)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim, device=devices)

        self.models = util.load_models_list(devices=cfgs.devices, tags=cfgs.data_tag)
        self.setEntropy_from_csv()

    def setEntropy_from_csv(self):
        self.entropy_df = {}
        self.paths = util.get_csv_path()

        for path in self.paths:
            self.entropy_df[path] = pd.read_csv(path, dtype=cfgs.entropy_column_dtypes)

    def get_Pretrained_Result(self, data, mode):
        score = pd.DataFrame(columns=cfgs.strModels)
        rank = pd.DataFrame(columns=cfgs.strModels)

        for strModel in cfgs.strModels:
            _score = pd.DataFrame(
                self.models[strModel].predict(data),
                columns=["score"],
            )

            score[strModel] = util.min_max_normalize(
                _score, "score", cfgs.normal_min, cfgs.normal_max
            )["score"]

            rank[strModel] = pd.DataFrame(
                [score[strModel].rank(ascending=True).iloc[cfgs.ground[mode]]]
            )

        return score, rank

    def set_min_val(self, df, idx):
        # Find the minimum value in the row
        min_value = df.iloc[idx].min()

        # Find all columns where the value matches the minimum
        min_columns = df.columns[df.iloc[idx] == min_value]

        for min_column in min_columns:
            cfgs.GROUND_SCORE[min_column] += 1

    def set_min_rank(self, ranks):
        min_value = ranks.min().min()
        min_columns = ranks.columns[ranks.isin([min_value]).any()]

        max_value = ranks.max().max()
        max_columns = ranks.columns[ranks.isin([max_value]).any()]

        if len(min_columns) < 4:
            for min_column in min_columns:
                cfgs.GROUND_RANK[min_column] += 1

            for max_column in max_columns:
                cfgs.GROUND_RANK_DOWN[max_column] += 1

    def _calc(self, data, r, mode):
        # util.endl(f"{mode}: {e}, {r}")
        pre_scores, ranks = self.get_Pretrained_Result(data, mode)

        # if cfgs.EVALUATION_ENTROPY_HOLD:
        #     entropy_score = self.get_entropy_score(r, "tail_batch")

        # else:
        entropy_score = self.get_entropy_score(r, mode)

        self.set_min_rank(ranks)
        self.set_min_val(pre_scores, cfgs.ground[mode])

        # cfgs.GROUND_SCORE[pre_scores.iloc[cfgs.ground[mode]].idxmin()] += 1

        # if cfgs.num_count_threshold > -2:
        if cfgs.num_count_threshold == -2:
            row_sums = pre_scores[
                pre_scores.iloc[cfgs.ground[mode]].idxmin()
            ].to_frame()
        elif entropy_score is None or cfgs.num_count_threshold == -1:
            row_sums = pre_scores
        elif entropy_score is not None:
            data = entropy_score[["model", cfgs.types_of_entropy]]

            if "TB" in cfgs.MODE_EVAL_NORM:
                mins, maxs = self.get_entropy_score_min_max(mode, cfgs.types_of_entropy)

                mm = pd.DataFrame(
                    [["mins", mins], ["maxs", maxs]],
                    columns=["model", cfgs.types_of_entropy],
                )
                data = pd.concat(
                    [mm, data],
                    axis=0,
                )

            # if cfgs.reverse_flag:
            #     normalize_entropy_score = util.min_max_normalize_reverse(
            #         data,
            #         cfgs.types_of_entropy,
            #         cfgs.entropy_normal_min,
            #         cfgs.entropy_normal_max,
            #     )

            # else:
            normalize_entropy_score = util.normalize[cfgs.MODE_EVAL_NORM](
                data,
                cfgs.types_of_entropy,
                cfgs.entropy_normal_min,
                cfgs.entropy_normal_max,
                reverse_Flag=cfgs.reverse_flag,
            )

            # print(entropy_score[["model", cfgs.types_of_entropy]])
            # print(data)

            # print(
            #     util.normalize_sigmoid(
            #         entropy_score[["model", cfgs.types_of_entropy]],
            #         cfgs.types_of_entropy,
            #         cfgs.entropy_normal_min,
            #         cfgs.entropy_normal_max,
            #         1.0,
            #     )
            # )
            # print(
            #     util.normalize_sigmoid(
            #         entropy_score[["model", cfgs.types_of_entropy]],
            #         cfgs.types_of_entropy,
            #         cfgs.entropy_normal_min,
            #         cfgs.entropy_normal_max,
            #         -1.0,
            #     )
            # )

            normalize_entropy_score.rename(
                columns={cfgs.types_of_entropy: "entropy_value"}, inplace=True
            )

            if cfgs.num_count_threshold >= 0:
                if cfgs.MODE_EVALUATION_TOP:
                    row_sums = self.calculate_weighted_resource(
                        pre_scores, normalize_entropy_score, True
                    )

                else:
                    row_sums = self.calculate_weighted_resource(
                        pre_scores, normalize_entropy_score
                    )

                # row_sums = self.calculate_weighted_resource(pre_scores, normalize_entropy_score).sum(axis=1)

                if cfgs.WRITE_EVAL_RESULT:
                    nums = self.get_entropy_num(r, mode)

                    rr = pd.DataFrame([r], columns=["relation"])

                    e_score = pd.concat(
                        [
                            rr,
                            nums,
                            ranks,
                            normalize_entropy_score.set_index("model")
                            .T.rename(columns=lambda x: "ent_" + x)
                            .reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    util.to_csv_Entropy(
                        e_score,
                        f"{cfgs.PATH_EVAL_RESULT}{cfgs.MODE_EVALUATION}/{mode}",
                        f"{cfgs.types_of_entropy}.csv",
                    )

            # elif cfgs.num_count_threshold == -1:
            #     # row_sums = self.calculate_weighted_resource(pre_scores, normalize_entropy_score, True).sum(axis=1)
            #     row_sums = pre_scores

            # if r == 13 and "c_" in cfgs.types_of_entropy and mode == "tail_batch":
            #     print(entropy_score[["model", cfgs.types_of_entropy]])
            #     print(normalize_entropy_score)
            #     print(self.calculate_weighted_resource(pre_scores, normalize_entropy_score))
            #     input()

        # print(row_sums)
        # input()

        result = torch.tensor(row_sums.sum(axis=1).values, dtype=torch.float32)

        if cfgs.devices != result.device:
            result = result.to(cfgs.devices)

        return result

    def calculate_weighted_resource(self, resource, weight, choose_top=False):
        """
        Multiplies each column of the `resource` DataFrame by its corresponding
        weight from the `weight` DataFrame, and returns a new DataFrame `result`.

        Parameters:
            resource (pd.DataFrame): The resource data.
            weight (pd.DataFrame): The weight data.

        Returns:
            pd.DataFrame: The weighted resource data.
        """

        result = pd.DataFrame()

        if choose_top:
            model = weight.loc[weight["entropy_value"].idxmin(), "model"]
            result[model] = resource[model]

        else:
            if weight is not None:
                for _, row in weight.iterrows():
                    model = row["model"]
                    entropy_value = row["entropy_value"]

                    if model in resource.columns and entropy_value > 0:
                        result[model] = resource[model] * entropy_value

                    # else:
                    # print(resource, weight)

        # print(resource)
        # print(weight)
        # print(result)
        # input()

        return result

    def get_entropy_score(self, r, mode, path=None):
        if cfgs.num_count_threshold == 0:
            return None

        # print(cfgs.num_count_threshold)
        # input()

        path = cfgs.entropy_path_id_short.replace("Tag", mode)

        _data = self.entropy_df[path][self.entropy_df[path]["relation"] == r]

        data = _data[_data["num_ori"] >= cfgs.num_count_threshold]

        # print(data)
        # input()

        if len(data) == 0:
            return None

        return data

    def get_entropy_score_min_max(self, mode, type_ent, path=None):
        if cfgs.num_count_threshold == 0:
            return None

        # entropy_score[["model", cfgs.types_of_entropy]]
        path = cfgs.entropy_path_id_short.replace("Tag", mode)

        data = self.entropy_df[path][[type_ent]]

        mins = data[data > 0].min().item()
        maxs = data.max().item()

        return mins, maxs

    def get_entropy_num(self, r, mode, path=None):
        if cfgs.num_count_threshold == 0:
            return None

        path = cfgs.entropy_path_id_short.replace("Tag", mode)

        _data = self.entropy_df[path][self.entropy_df[path]["relation"] == r]

        result = _data["num_ori"].to_frame().T
        result.columns = ["n_" + item for item in cfgs.strModels]
        result.reset_index(drop=True, inplace=True)

        return result

    def forward(self, data):
        # batch_h = data["batch_h"]
        # batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        mode = data["mode"]

        score = self._calc(data, batch_r.item(), mode)

        return score

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()

    # def forward(self, data):
    #     # print("forward")
    #     batch_h = data["batch_h"]
    #     batch_t = data["batch_t"]
    #     batch_r = data["batch_r"]
    #     mode = data["mode"]

    #     h = self.ent_embeddings(batch_h)
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     score = self._calc(h, t, r, mode)
    #     if self.margin_flag:
    #         return self.margin - score
    #     else:
    #         return score
