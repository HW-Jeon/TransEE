import importlib
import math
import os.path
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial import distance
from scipy.stats import alpha, anderson, kstest, norm
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KernelDensity
from torch.distributions import Categorical
from torch.nn.functional import cosine_similarity

import config_TransEE as cfgs
import utilities as util
from PDF_Fitter import PDF_Fitters


class EntropyCalculator:
    def __init__(self):
        importlib.reload(util)

        self.tensor = None
        self.n_dist = None
        self.pairwise_sim = None
        self.mahalanobis_dist = None
        self.n_dist_p = None
        self.pairwise_sim_p = None
        self.mahalanobis_dist_p = None

        self.rounds = cfgs.rounds
        self.alpha = cfgs.alpha
        self.q = cfgs.q

        self.get_distance_probs = {
            "pdf": self.pdf_Fitting,
            "cdf": self.gaussian_pdf_using_scipy,
            "hist": self.get_prob_from_Histogram,
            "curve": self.fit_gaussian_and_get_probability,
        }

        self.get_distance_resources = {
            "euclidean": self.get_pairwise_distance,
            "cosine": self.get_pairwise_cosine_similarity,
            "mahalanobis": self.pairwise_mahalanobis_distance,
        }

        self.calculate_entropie = {
            "DEFAULT": self.calculate_entropies,
            "SCIPY": self.calculate_entropies,
            "SCIPY_NO_MAHA": self.calculate_entropies_no_maha,
            "DEFAULT_NO_MAHA": self.calculate_entropies_no_maha,
        }

        self.resource_attribute_map = {
            "euclidean": ("n_dist", "n_dist_p"),
            "cosine": ("pairwise_sim", "pairwise_sim_p"),
            "mahalanobis": ("mahalanobis_dist", "mahalanobis_dist_p"),
        }

        self.get_distance_prob = self.get_distance_probs[cfgs.CURRENT_ENTROPY_SELECTOR]

    def __del__(self):
        del self.tensor
        del self.n_dist
        del self.pairwise_sim
        del self.mahalanobis_dist

        torch.cuda.empty_cache()

    def set_resource(self, resource, minSize=0):
        if self.tensor is not None:
            del self.tensor
            del self.n_dist
            del self.pairwise_sim
            del self.mahalanobis_dist

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # print(
        #     f"probs: {cfgs.CURRENT_PAIRD} {cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
        # )

        # util.check_CUDA_MEM("calc")

        if "Not" not in cfgs.CURRENT_PAIRD:
            self.tensor = resource.to(cfgs.devices)[:minSize, :]

        else:
            self.tensor = resource.to(cfgs.devices)

        if torch.is_complex(self.tensor):
            self.tensor = torch.abs(self.tensor)

        # print("set")

        _tensor = self.tensor.detach().cpu()

        # self.fig, self.ax1 = plt.subplots(figsize=(16, 10))

        for ent_resource in cfgs.ENTROPY_RESOURCES:
            # Check if the ent_resource is in the mapping
            if ent_resource in self.resource_attribute_map:
                attr1, attr2 = self.resource_attribute_map[ent_resource]

                dist, pval = self.get_distance_resources[ent_resource](_tensor)

                # Dynamically set the attributes of self
                setattr(self, attr1, dist)
                setattr(self, attr2, pval)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # self.ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        # self.ax1.set_title(f"Probability Density Function: TransD, relation 14")
        # self.ax1.set_xlabel("Distance")
        # self.ax1.set_ylabel("Probability")
        # # self.ax1.legend(loc="lower right")
        # # self.ax1.legend()

        # self.fig.show()

        # input()

        plt.close()

        # PRINT()

        # print("set Done")

        # print(self.n_dist, self.n_dist_p)

        # if cfgs.CURRENT_MODEL == "TransR":
        #     print(f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}")
        #     print(self.pairwise_sim, self.pairwise_sim_p)
        #     # print(self.mahalanobis_dist, self.mahalanobis_dist_p)

        #     input()

        # self.n_dist, self.n_dist_p = self.get_pairwise_distance(self.tensor)

        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # self.pairwise_sim, self.pairwise_sim_p = self.get_pairwise_cosine_similarity(self.tensor)

        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # self.mahalanobis_dist, self.mahalanobis_dist_p = self.pairwise_mahalanobis_distance(self.tensor)

        # # torch.cuda.synchronize()
        # # torch.cuda.empty_cache()

        # print(self.n_dist, self.n_dist_p)
        # print(self.pairwise_sim, self.pairwise_sim_p)
        # print(self.mahalanobis_dist, self.mahalanobis_dist_p)

        del resource
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return

    def get_n(self):
        return self.tensor

    # def get_cosine(self):
    #     return self.pairwise_dist

    def get_cosine_sim(self):
        return self.pairwise_sim

    def gaussian(self, x, mean, amplitude, standard_deviation):
        return amplitude * np.exp(-(((x - mean) / standard_deviation) ** 2))

    def gaussian_2(self, x, mu, sigma):
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((x - mu) ** 2) / (2 * sigma**2)
        )

    def burr_pdf(self, x, c, k):
        """Burr distribution probability density function for curve fitting."""
        return c * k * x ** (c - 1) * (1 + x**c) ** (-k - 1)

    # Function to fit data to a Gaussian and compute the probability for each data point
    def fit_gaussian_and_get_probability(self, _tensor, fitter=stats.norm):
        try:
            if _tensor.is_cuda:
                tensor = _tensor.detach().cpu()

            else:
                tensor = _tensor

            # Convert tensor to numpy
            tensor_np = tensor.numpy()

            hist_values, bin_edges = util.getHistogram(tensor_np)

            printed_hist = hist_values * np.diff(bin_edges)
            printed_hist_edge = bin_edges

            if "mahalanobis" in cfgs.CURRENT_LABEL:
                # print(cfgs.CURRENT_LABEL)

                hist_values, bin_edges = util.getHistogram(
                    tensor_np[tensor_np > bin_edges[1]]
                )

                # hist_values = hist_values[1:]
                # bin_edges = bin_edges[1:]

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_values = hist_values * np.diff(bin_edges)

            if fitter == stats.norm:
                mu, sigma = norm.fit(tensor_np)
                amplitude = 1 / (sigma * np.sqrt(2 * np.pi))

                # Fit the Gaussian function to the histogram
                params, _ = curve_fit(
                    self.gaussian,
                    bin_centers,
                    hist_values,
                    bounds=(0, np.inf),
                    p0=[mu, amplitude, sigma],
                )
                prob_density = self.gaussian(tensor_np, *params)

                try:
                    _, p_value = kstest(tensor_np, "norm", args=(params[0], params[1]))

                except RuntimeWarning as e:
                    p_value = 1

                func_fit = self.gaussian

            elif fitter == stats.burr:
                print("is Burr")

                param = fitter.fit(tensor_np)
                params, _ = curve_fit(
                    self.burr_pdf, bin_centers, hist_values, p0=[param[0], param[1]]
                )
                prob_density = self.burr_pdf(tensor_np, *params)

                _, p_value = kstest(tensor_np, "burr", args=(params[0], params[1]))

                func_fit = self.burr_pdf

            if "IMAGE" in cfgs.MODE:
                # torch.save(tensor, "./tensor/tensor_tmp.pt")

                plt.subplots(figsize=(10, 6))
                x = np.linspace(tensor.min(), tensor.max(), 1000)

                # if "mahalanobis" in cfgs.CURRENT_LABEL:
                #     ax1.hist(
                #         bin_edges[:-1],
                #         bin_edges,
                #         alpha=0.6,
                #         color="b",
                #         label="Deleted Histogram",
                #         weights=Categorical(torch.tensor(hist_values)).probs.numpy(),
                #     )
                # else:
                #     ax1.hist(
                #         printed_hist_edge[:-1],
                #         printed_hist_edge,
                #         alpha=0.5,
                #         color="g",
                #         label="Histogram of tensor",
                #         weights=Categorical(torch.tensor(hist_values)).probs.numpy(),
                #     )

                param = fitter.fit(tensor_np)

                plt.plot(
                    tensor_np,
                    Categorical(
                        torch.tensor(fitter.pdf(tensor_np, *param))
                    ).probs.numpy(),
                    "r",
                    label="Fitted Function",
                )
                plt.scatter(
                    tensor_np,
                    Categorical(torch.tensor(prob_density)).probs.numpy(),
                    color="black",
                    marker="x",
                    s=10,
                    label="Data Points",
                )

                plt.title(
                    f"Fitted Gaussian {cfgs.CURRENT_REL} {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor_np.size} {hist_values.size}"
                )
                plt.xlabel("Value")
                plt.legend()

                if "SHOW" in cfgs.MODE:
                    plt.show()

                if "SAVE" in cfgs.MODE:
                    util.makedirs(
                        f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}"
                    )

                    plt.savefig(
                        f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                        format="png",
                    )

                plt.close()

            del _tensor, tensor
            torch.cuda.empty_cache()

            return torch.tensor(prob_density, device=cfgs.devices), p_value

        except RuntimeError as e:
            print(e)
            print(
                f"Runtime Error Handling: {cfgs.CURRENT_PAIRD} {cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )

            if _tensor.is_cuda:
                tensor = _tensor.detach().cpu()

            else:
                tensor = _tensor

            # Convert tensor to numpy
            tensor_np = tensor.numpy()

            hist_values, bin_edges = util.getHistogram(tensor_np)

            if "mahalanobis" in cfgs.CURRENT_LABEL:
                hist_values, bin_edges = util.getHistogram(
                    tensor_np[tensor_np > bin_edges[1]]
                )

            hist_values = hist_values * np.diff(bin_edges)

            prob_density = util.get_hist_data(tensor_np, hist_values, bin_edges)

            mu, sigma = norm.fit(tensor)

            _, p_value = kstest(tensor_np, "norm", args=(mu, sigma))

            # torch.save(tensor, "./tensor/tensor_tmp.pt")

            fig, ax1 = plt.subplots(figsize=(10, 6))
            # x = np.linspace(tensor.min(), tensor.max(), 1000)
            # ax1.hist(printed_hist_edge[:-1], printed_hist_edge, alpha=0.6, color="g", label="Histogram of tensor", weights=printed_hist)
            ax1.scatter(
                tensor_np,
                prob_density,
                color="black",
                marker="x",
                s=10,
                label="Data Points",
            )

            plt.title(
                f"Fitted Gaussian {cfgs.CURRENT_REL} {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor_np.size} {hist_values.size}"
            )
            plt.xlabel("Value")
            plt.legend()

            if "SHOW" in cfgs.MODE:
                plt.show()

            if "SAVE" in cfgs.MODE:
                util.makedirs(
                    f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}"
                )

                plt.savefig(
                    f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                    format="png",
                )
            plt.close()

            return torch.tensor(prob_density, device=cfgs.devices), p_value

    def get_prob_from_Histogram(self, _tensor, fitter=stats.norm):
        if _tensor.is_cuda:
            tensor = _tensor.detach().cpu()

        else:
            tensor = _tensor

        # Convert tensor to numpy
        tensor_np = tensor.numpy()

        hist_values, bin_edges = util.getHistogram(tensor_np)

        ################################
        # 위치 중요
        hist_values = hist_values * np.diff(bin_edges)

        # print("Sum: ", hist_values.sum())

        prob_density = util.get_hist_data(tensor_np, hist_values, bin_edges)

        ################################

        if "mahalanobis" in cfgs.CURRENT_LABEL:
            hist_values, bin_edges = util.getHistogram(
                tensor_np[tensor_np > bin_edges[1]]
            )

        ################################
        # 여기가 맞나?
        ###############################

        mu, sigma = norm.fit(tensor)

        _, p_value = kstest(
            tensor_np[tensor_np > bin_edges[1]], "norm", args=(mu, sigma)
        )

        # print("Hist Min: ", prob_density.min())
        # print("Hist Max: ", prob_density.max())

        # plt.figure(figsize=(8, 6))

        # plt.plot(prob_density)
        # plt.show()
        # plt.close()

        # torch.save(tensor, "./tensor/tensor_tmp.pt")
        if "IMAGE" in cfgs.MODE:
            plt.figure(figsize=(8, 6))
            # x = np.linspace(tensor.min(), tensor.max(), 1000)
            # ax1.hist(printed_hist_edge[:-1], printed_hist_edge, alpha=0.6, color="g", label="Histogram of tensor", weights=printed_hist)
            plt.scatter(
                tensor_np,
                prob_density,
                color="black",
                marker="x",
                s=10,
                label="Data Points",
            )

            plt.title(
                f"Fitted Gaussian {cfgs.CURRENT_REL} {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor_np.size} {hist_values.size}"
            )
            plt.xlabel("Value")
            plt.legend()

            if "SHOW" in cfgs.MODE:
                plt.show()

            if "SAVE" in cfgs.MODE:
                util.makedirs(
                    f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}"
                )

                plt.savefig(
                    f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                    format="png",
                )
            plt.close()

        return torch.tensor(prob_density, device=cfgs.devices), p_value

    def gaussian_pdf_using_scipy(self, _tensor, fitter=norm):
        """Calculate the Gaussian PDF of the tensor using scipy.stats"""

        if _tensor.is_cuda:
            tensor = _tensor.detach().cpu()

        else:
            tensor = _tensor

        delta = ((tensor.max() - tensor.min()) / len(tensor)).item()

        sorted_np = np.unique(np.sort(tensor))

        params = fitter.fit(sorted_np)

        # Calculate the Gaussian PDF
        pdf_values = fitter.pdf(tensor, *params)

        # Calculate approximate probabilities

        # diff = np.diff(sorted_np)
        # diff = np.min(diff[np.nonzero(diff)])

        probs_down = fitter.cdf(tensor - (delta * 0.75), *params)
        probs_up = fitter.cdf(tensor + (delta * 0.75), *params)

        probabilities = probs_up - probs_down

        # _, ks_pvalue = kstest(sorted_np, "norm", args=(mu, sigma))

        ks_pvalue = 0.0

        # print("probabilities min: ", probabilities.min())
        # print("probabilities max: ", probabilities.max())
        # print("probabilities sum: ", probabilities.sum())
        # # print("diff - ", diff)
        # print("delta - ", delta)

        # util.plots_pdf_values(sorted_np, probabilities, fitter.cdf(sorted_np, loc=mu, scale=sigma))
        def plots_pdf_values(tensor, probs, pdf_values):
            maxs = probs.max().item()
            pdf_max = pdf_values.max().item()

            print(maxs)
            print(pdf_max)
            print(maxs / pdf_max)

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
                x=tensor, y=probs, color="green", marker="o", s=5, label="Probs"
            )

            plt.scatter(
                x=tensor,
                y=pdf_values * maxs / pdf_max,
                color="red",
                marker="x",
                s=5,
                label="PDF",
            )
            plt.ylim([maxs * -0.2, maxs * 1.1])

            plt.title(
                f"Fitted Gaussian CDF {cfgs.CURRENT_REL} {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor.shape[0]}"
            )
            plt.xlabel("Value")
            plt.legend()

            if "SHOW" in cfgs.MODE:
                plt.show()

            if "SAVE" in cfgs.MODE:
                util.makedirs(
                    f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}"
                )

                plt.savefig(
                    f"./img_hist/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                    format="png",
                )
            plt.close()

        if "IMAGE" in cfgs.MODE:
            # plots_pdf_values(tensor, probabilities, fitter.cdf(tensor, *params))
            plots_pdf_values(
                tensor,
                Categorical(torch.tensor(probabilities)).probs,
                Categorical(torch.tensor(pdf_values)).probs,
            )

        # util.plot_gaussian(tensor, pdf_values)

        del _tensor, tensor, pdf_values, delta, probs_down, probs_up
        torch.cuda.empty_cache()

        return torch.tensor(probabilities, device="cuda:0"), ks_pvalue

    def pdf_Fitting(self, _tensor, fitter=norm, strFitter="norm"):
        """Calculate the Gaussian PDF of the tensor using scipy.stats"""

        # util.endl_time("pdf_Fitting start")

        # util.print_entropy_header()
        try:
            if _tensor.is_cuda:
                # tensor = _tensor.detach().cpu()
                tensor = util.resize_tensor(
                    _tensor.detach().cpu(), cfgs.MODE_MIN_RESOURCE
                )

            else:
                tensor = util.resize_tensor(_tensor, cfgs.MODE_MIN_RESOURCE)

                #     input()

            # util.endl_time("pdf_Fitting fit init")
            # print(f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}")

            if "SCIPY" in cfgs.CALC_MODE:
                tensor = torch.softmax(tensor, dim=0)

            params = fitter.fit(tensor)
            # util.endl_time("pdf_Fitting fit")

            # Calculate the Gaussian PDF
            pdf_values = fitter.pdf(tensor, *params)

            # util.endl_time("pdf_Fitting pdf")

            # ks_pvalue = 0.0

            try:
                # _, p_value = kstest(tensor, strFitter, args=(params[0], params[1]))
                _, p_value = kstest(tensor, fitter.cdf(tensor, *params))

            except RuntimeWarning as e:
                p_value = 1
            # util.endl_time("pdf_Fitting ks")

            # util.plots_pdf_values(sorted_np, probabilities, fitter.cdf(sorted_np, loc=mu, scale=sigma))
            def plots_pdf_values(tensor, probs, limit_img=0):
                maxs = probs.max().item()

                hist_values, hist_edge = np.histogram(tensor, bins="auto", density=True)

                plt.figure(figsize=(8, 6))

                plt.hist(
                    hist_edge[:-1],
                    hist_edge,
                    alpha=0.6,
                    color="b",
                    label="Histogram of tensor",
                    weights=hist_values * maxs / hist_values.max(),
                )

                plt.scatter(
                    x=tensor, y=probs, color="green", marker="o", s=5, label="Probs"
                )
                plt.ylim([maxs * -0.2, maxs * 1.1])

                plt.title(
                    f"Fitted Gaussian CDF {cfgs.CURRENT_REL} {cfgs.CURRENT_MODEL} {cfgs.CURRENT_LABEL} {tensor.shape[0]}"
                )
                plt.xlabel("Value")
                plt.legend()

                # plt.show()

                # print("File name:")
                # print(
                #     f"./img_hist/{cfgs.dataset}/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/Resize/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                # )

                if "SHOW" in cfgs.MODE:
                    plt.show()

                if "SAVE" in cfgs.MODE:
                    util.makedirs(
                        f"./img_hist/{cfgs.dataset}/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}"
                    )

                    plt.savefig(
                        f"./img_hist/{cfgs.dataset}/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                        format="png",
                    )

                    if limit_img > 0:
                        util.makedirs(
                            f"./img_hist/{cfgs.dataset}/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/Resize/"
                        )

                        plt.savefig(
                            f"./img_hist/{cfgs.dataset}/{cfgs.CURRENT_PAIRD}/{cfgs.CURRENT_BATCH}/{cfgs.CURRENT_ENTROPY_SELECTOR}/Resize/{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}.png",
                            format="png",
                        )

                plt.close()

            # if _tensor.shape[0] > cfgs.MODE_MIN_RESOURCE:
            #     print(
            #         f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            #     )
            #     print(_tensor.shape)
            #     print(tensor.shape)

            if "PDF" in cfgs.MODE and "maha" not in cfgs.CURRENT_LABEL:
                pdf_tensor = torch.tensor(pdf_values)

                util.comp_plots_pdf_values(
                    tensor[~torch.isnan(pdf_tensor)],
                    Categorical(pdf_tensor[~torch.isnan(pdf_tensor)]).probs,
                    self.fig,
                    self.ax1,
                )

            if "IMAGE" in cfgs.MODE:
                # plots_pdf_values(tensor, probabilities, fitter.cdf(tensor, *params))

                pdf_tensor = torch.tensor(pdf_values)

                plots_pdf_values(
                    tensor[~torch.isnan(pdf_tensor)],
                    Categorical(pdf_tensor[~torch.isnan(pdf_tensor)]).probs,
                    int(_tensor.shape[0] - tensor.shape[0]),
                )

                del pdf_tensor

            # util.plot_gaussian(tensor, pdf_values)

            if "DEFAULT" in cfgs.CALC_MODE:
                result_tmp = torch.tensor(pdf_values, device="cuda:0").unsqueeze(0)

                result = result_tmp[~torch.isnan(result_tmp)].reshape(1, -1)
                del result_tmp

            else:
                result = fitter.entropy(*params)

            del _tensor, tensor
            torch.cuda.empty_cache()

            return result, p_value

            # filtered_tensor = tensor[~torch.isnan(tensor)].reshape(1, -1)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("pdf_Fitting RuntimeError, CUDA out of memory")
                ts_cpu = tensor.detach().cpu()

                distance, p_value = self.pdf_Fitting(
                    ts_cpu, fitter=fitter, strFitter=strFitter
                )
                print("CUDA out of memory handling")
                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                return distance, p_value
            else:
                print("pdf_Fitting RuntimeError", e)
            return math.nan
        except Exception as e:
            print("pdf_Fitting Exception", e)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return math.nan, None

    def compute_distribution(self, tensor, bins="auto"):
        """
        Compute distribution (histogram) of a tensor.

        Args:
        - tensor (torch.Tensor): Input tensor of shape [n].
        - bins (str or int or list): Number of bins, "auto", or a list of bin edges.

        Returns:
        - distribution (torch.Tensor): Tensor containing the histogram distribution.
        - bin_edges (torch.Tensor): Tensor containing the bin edges.
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

        # Compute histogram
        hist = torch.histc(
            tensor, bins=bins, min=float(tensor.min()), max=float(tensor.max())
        )

        # Compute bin edges
        bin_edges = torch.linspace(tensor.min().item(), tensor.max().item(), bins + 1)

        # print(bins)
        # print(hist)

        return hist, bin_edges

    def pairwise_mahalanobis_distance(self, tensor):
        try:
            with torch.no_grad():
                cfgs.CURRENT_LABEL = "mahalanobis"

                # Compute the covariance and its inverse
                mean_tensor = tensor.mean(dim=0)
                centered_tensor = tensor - mean_tensor
                cov = torch.mm(centered_tensor.t(), centered_tensor) / (
                    tensor.size(0) - 1
                )
                V = torch.inverse(cov)

                # Compute pairwise differences using broadcasting
                diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)

                # Compute pairwise Mahalanobis distances
                mahalanobis_sq = torch.einsum("ijk,kl,ijl->ij", diff, V, diff)

                del V, diff, cov, centered_tensor, mean_tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Ensure the result is non-negative (there can be small negative values due to numerical precision issues)
                mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0)

                # Get the upper triangle of the distance matrix without the diagonal
                upper_tri_indices = torch.triu_indices(
                    tensor.size(0), tensor.size(0), offset=1, device=tensor.device
                )
                # result = torch.sqrt(mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]])
                # result, _ = self.compute_distribution(torch.sqrt(mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]]))

                # pdf_Fitter = PDF_Fitters()

                # tmp, _ = self.get_prob_from_Histogram(torch.sqrt(mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]]))

                result, p_value = self.get_distance_prob(
                    torch.sqrt(
                        mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]]
                    )
                )

                # result, p_value = self.fit_gaussian_and_get_probability(torch.sqrt(mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]]))

                # if cfgs.MAHALANOBIS_SLICE:
                #     # return result[result >= 1].unsqueeze(0), p_value
                #     result, p_value = self.fit_gaussian_and_get_probability(
                #         torch.sqrt(mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]]), False
                #     )

                # else:
                #     result, p_value = self.fit_gaussian_and_get_probability(torch.sqrt(mahalanobis_sq[upper_tri_indices[0], upper_tri_indices[1]]))

                # Delete local tensors to potentially free up memory faster

            del tensor, mahalanobis_sq, upper_tri_indices
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Optional: Clear CUDA memory cache

            return result, p_value
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("pairwise_mahalanobis_distance RuntimeError, CUDA out of memory")
                ts_cpu = tensor.detach().cpu()

                distance, p_value = self.pairwise_mahalanobis_distance(ts_cpu)
                print("CUDA out of memory handling")
                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                return distance, p_value
            else:
                print("pairwise_mahalanobis_distance RuntimeError", e)
            return math.nan
        except Exception as e:
            print("pairwise_mahalanobis_distance Exception", e)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan, None

    def torch_cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=1)

        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())

        return cov

    def get_pairwise_distance(self, tensor: torch.Tensor):
        # Compute pairwise distances using matrix multiplication

        cfgs.CURRENT_LABEL = "Euclidean"

        # print(tensor.shape)

        norms = (tensor**2).sum(dim=-1).view(-1, 1)
        _dists_sq = norms - 2.0 * torch.mm(tensor, tensor.t()) + norms.t()

        # Ensure the result is non-negative (there can be small negative values due to numerical precision issues)
        dists_sq = torch.clamp(_dists_sq, min=0.0)

        # Take the square root to get actual distances
        dists = torch.sqrt(dists_sq)

        # Get the upper triangle of the distance matrix without the diagonal
        upper_tri_indices = torch.triu_indices(
            tensor.size(0), tensor.size(0), offset=1, device=tensor.device
        )
        # result, _ = self.compute_distribution(dists[upper_tri_indices[0], upper_tri_indices[1]])
        # result = dists[upper_tri_indices[0], upper_tri_indices[1]]

        # print(dists[upper_tri_indices[0], upper_tri_indices[1]].shape)

        # util.check_CUDA_MEM("get_pairwise_distance 1")

        del tensor, norms, _dists_sq, dists_sq
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # util.check_CUDA_MEM("get_pairwise_distance 2")

        # pdf_Fitter = PDF_Fitters()

        # print(1 / dists.shape[0])

        result, p_value = self.get_distance_prob(
            dists[upper_tri_indices[0], upper_tri_indices[1]],
            fitter=stats.burr,
            strFitter="burr",
        )

        # data = dists[upper_tri_indices[0], upper_tri_indices[1]].detach().cpu()

        # t1, _ = self.pdf_Fitting(data, fitter=stats.burr)
        # t2, _ = self.gaussian_pdf_using_scipy(data, fitter=stats.burr)
        # t3, _ = self.get_prob_from_Histogram(data, fitter=stats.burr)
        # t4, _ = self.fit_gaussian_and_get_probability(data, fitter=stats.burr)

        # plt.figure(figsize=(8, 6))

        # axis_s = Categorical(t1).probs.detach().cpu().numpy().max()

        # plt.scatter(x=data, y=Categorical(t1).probs.detach().cpu().numpy(), color="blue", marker="x", s=3, label="pdf")
        # plt.scatter(x=data, y=Categorical(t2).probs.detach().cpu().numpy(), color="red", marker="x", s=3, label="cdf")
        # plt.scatter(x=data, y=Categorical(t3).probs.detach().cpu().numpy(), color="yellow", marker="x", s=3, label="hist")
        # plt.scatter(x=data, y=Categorical(t4).probs.detach().cpu().numpy(), color="green", marker="x", s=3, label="curve")

        # plt.ylim([axis_s * -0.1, axis_s * 1.1])

        # plt.show()
        # plt.close()

        # result, p_value = self.gaussian_pdf_using_scipy(dists[upper_tri_indices[0], upper_tri_indices[1]], fitter=stats.burr)

        # tmp, _ = self.fit_gaussian_and_get_probability(dists[upper_tri_indices[0], upper_tri_indices[1]], fitter=stats.burr)

        # result, p_value = self.gaussian_pdf_using_scipy(dists[upper_tri_indices[0], upper_tri_indices[1]], fitter=stats.burr)

        # tensor = dists[upper_tri_indices[0], upper_tri_indices[1]]
        # tensor_np = tensor.detach().cpu().numpy()

        # tmp = Categorical(tmp).probs.detach().cpu().numpy()

        # plt.figure(figsize=(8, 6))
        # plt.scatter(x=tensor_np, y=tmp, color="red", marker="x", s=10, label="Histogram")
        # plt.scatter(x=tensor_np, y=Categorical(result).probs.detach().cpu().numpy(), color="blue", marker="x", s=10, label="CDF")
        # plt.ylim([tmp.min() * 0.9, tmp.max() * 1.1])
        # plt.show()

        # plt.close()

        # Delete local tensors to potentially free up memory faster
        del dists, upper_tri_indices

        # Optional: Clear CUDA memory cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return result, p_value

    def get_pairwise_cosine_similarity(self, tensor):
        cfgs.CURRENT_LABEL = "cosine"

        # L2 normalize the tensor along the feature dimension
        tensor_norm = torch.nn.functional.normalize(tensor, p=2, dim=1)

        # Compute cosine similarities using matrix multiplication
        cos_sim = torch.mm(tensor_norm, tensor_norm.t())

        # Get the upper triangle of the matrix without the diagonal
        upper_tri_indices = torch.triu_indices(
            tensor.size(0), tensor.size(0), offset=1, device=tensor.device
        )
        # result = cos_sim[upper_tri_indices[0], upper_tri_indices[1]]
        # result, _ = self.compute_distribution(cos_sim[upper_tri_indices[0], upper_tri_indices[1]])

        cos_sims = cos_sim[upper_tri_indices[0], upper_tri_indices[1]]

        del tensor, tensor_norm, cos_sim, upper_tri_indices
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # cos_sim_result, p_value = self.get_prob_from_Histogram(cos_sims)

        cos_sim_result, p_value = self.get_distance_prob(
            cos_sims, fitter=stats.burr, strFitter="burr"
        )

        # cos_sim_result, p_value = self.gaussian_pdf_using_scipy(cos_sims, fitter=stats.burr)
        # cos_sim_result, p_value = self.fit_gaussian_and_get_probability(cos_sims)

        del cos_sims
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # print(result)
        # print(result.shape)
        # print(torch.min(result))
        # print(torch.max(result))
        # print(torch.sum(result))

        # probs = torch.softmax(result.unsqueeze(0), dim=len(tensor.shape) - 1)

        # input()

        # print(probs)
        # print(probs.shape)
        # print(torch.min(probs))
        # print(torch.max(probs))
        # print(torch.sum(probs))

        # print(self.shannon_entropy(result.unsqueeze(0)))
        # print(self.shannon_entropy((1 - result).unsqueeze(0)))
        # print(self.shannon_entropy(1 - (result.unsqueeze(0))))

        # input()

        # Delete local tensors to potentially free up memory faster

        # Optional: Clear CUDA memory cache

        return cos_sim_result, p_value

    # def get_pairwise_cosine_distance(self, tensor):
    #     return 1 - self.get_pairwise_cosine_similarity(tensor)

    def pairwise_cosine_similarity(self):
        try:
            if torch.is_complex(self.tensor):
                norm_tensor = torch.abs(self.tensor)
            else:
                norm_tensor = self.tensor

            # Compute pairwise cosine similarities
            norms = torch.norm(norm_tensor, dim=1, keepdim=True)
            cosine_similarities = torch.mm(self.tensor, self.tensor.t()) / torch.mm(
                norms, norms.t()
            )

            rows, cols = cosine_similarities.shape

            mask = ~(torch.arange(rows).unsqueeze(1) == torch.arange(cols).unsqueeze(0))

            # Get non-diagonal elements using the mask
            non_diag = cosine_similarities[mask].reshape(rows, cols - 1)

            return non_diag.to(device=cfgs.devices)
        except Exception as e:
            print(e)
            return 0.00

    def pairwise_cosine_distance(self):
        if torch.is_complex(self.tensor):
            norm_tensor = torch.abs(self.tensor)
        else:
            norm_tensor = self.tensor

        # Normalize each vector (row) in the tensor
        norm = torch.norm(norm_tensor, dim=1, keepdim=True)
        normalized_tensor = self.tensor / norm

        # Compute cosine similarity
        cosine_similarity = torch.mm(normalized_tensor, normalized_tensor.t())

        # Convert cosine similarity to cosine distance
        cosine_distance = 1 - cosine_similarity

        rows, cols = cosine_distance.shape

        mask = ~(torch.arange(rows).unsqueeze(1) == torch.arange(cols).unsqueeze(0))

        # Get non-diagonal elements using the mask
        non_diag = cosine_distance[mask].reshape(rows, cols - 1)

        return non_diag.to(device=cfgs.devices)

    def get_logits(self, logit):
        min_real = torch.finfo(logit.dtype).min
        logits = torch.clamp(logit, min=min_real)

        del logit, min_real

        torch.cuda.empty_cache()

        return logits

    def shannon_entropy(self, tensor):
        try:
            # util.endl_time("shannon_entropy start")

            # probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor

            # Categorical same
            # Categorical(probs=probs).entropy()
            # entropy = -torch.sum(probabilities * torch.log2(probabilities))

            # print(Categorical(probs=probs).probs.sum())

            # print(torch.softmax(tensor, dim=len(tensor.shape) - 1).sum())

            # print((Categorical(probs=probs).probs - torch.softmax(tensor, dim=len(tensor.shape) - 1)).sum())

            # if cfgs.CURRENT_MODEL == "TransR":
            #     print(torch.softmax(tensor, dim=len(tensor.shape) - 1))
            #     print((Categorical(probs=probs).probs))
            #     input()

            entropy = Categorical(probs=probs).entropy().mean().item()

            del probs
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # util.endl_time("shannon_entropy")

            return entropy
        except RuntimeError as e:
            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )

            if "CUDA out of memory" in str(e):
                print("Shannon_entropy CUDA out of memory")
                print(tensor.shape)
                ts_cpu = tensor.detach().cpu()
                entropy = self.shannon_entropy(ts_cpu)
                print("CUDA out of memory handling")

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return entropy
            else:
                print("Shannon_entropy ", e)
                print(tensor.shape)
                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return math.nan
        except Exception as e:
            if "invalid values" in str(e):
                filtered_tensor = tensor[~torch.isnan(tensor)].reshape(1, -1)

                probs = torch.softmax(
                    filtered_tensor, dim=len(filtered_tensor.shape) - 1
                )

                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                # print("new entropy")
                # print(entropy)

                # input()

                del tensor, filtered_tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                util.endl(str(entropy))

                return entropy

            print("Shannon_entropy ", e)

            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def conditional_entropy_2(self, tensor):
        try:
            # Ensure the tensor sums up to 1 along the last dimension
            tensor /= tensor.sum(dim=-1, keepdim=True)

            # Create a categorical distribution
            dist = Categorical(probs=tensor)

            # Compute entropy for each distribution in the tensor
            entropies = dist.entropy()

            # Weight each entropy by its probability to get conditional entropy
            entropy = (tensor * entropies).sum(dim=-1)

            # print(entropy)
            # input()

            del tensor, dist, entropies

            return entropy.item()

        except RuntimeError as e:
            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("Conditional_entropy_ RuntimeError ", e)

            if "CUDA out of memory" in str(e):
                print(tensor.shape)
                ts_cpu = tensor.detach().cpu()
                entropy = self.conditional_entropy_(ts_cpu)
                print("CUDA out of memory handling")

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return entropy
            return math.nan
        except Exception as e:
            print("Conditional_entropy_ ", e)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def conditional_entropy_(self, tensor):
        try:
            # joint_probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            # joint_probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)

            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor

            probs_logits = Categorical(probs=tensor).logits()
            min_real = torch.finfo(probs_logits.dtype).min
            logits = torch.clamp(probs_logits, min=min_real)

            # probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)

            # Marginalize probs over Y to get p(x)
            p_x = torch.sum(probs, dim=1)

            # Compute p(y|x) = p(x, y) / p(x) for all x and y
            p_y_given_x = probs / p_x[:, None]

            # Calculate entropy H(Y|X=x) for all x and replace NaNs with 0
            h_y_given_x = -torch.sum(
                p_y_given_x * torch.log2(p_y_given_x + 1e-9), dim=1
            )
            h_y_given_x[torch.isnan(h_y_given_x)] = 0

            # Compute H(Y|X) = sum_x p(x) * H(Y|X=x)
            h_y_given_xs = torch.sum(p_x * h_y_given_x)

            del tensor, p_x, p_y_given_x, h_y_given_x, probs_logits, min_real, logits

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            return h_y_given_xs.item()

        except RuntimeError as e:
            print("Conditional_entropy_ RuntimeError ", e)

            if "CUDA out of memory" in str(e):
                print(tensor.shape)
                ts_cpu = tensor.detach().cpu()
                entropy = self.conditional_entropy_(ts_cpu)
                print("CUDA out of memory handling")

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return entropy
            return math.nan
        except Exception as e:
            print("Conditional_entropy_ ", e)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def conditional_entropy(self, tensor):
        try:
            # probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor

            # Shannon_entropy
            shannon_entropy = Categorical(probs=probs).entropy().mean()

            # marginal_entropy = Categorical(probs=probs.mean(dim=0)).entropy()

            # Compute marginal entropy based on tensor's dimensions
            marginal_entropy = Categorical(probs=probs.mean(dim=0)).entropy()

            entropy = (shannon_entropy - marginal_entropy).item()

            del probs
            del tensor
            del marginal_entropy
            del marginal_probs
            del shannon_entropy
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            return entropy
        except RuntimeError as e:
            print("Conditional_entropy RuntimeError ", e)

            if "CUDA out of memory" in str(e):
                print(tensor.shape)
                ts_cpu = tensor.detach().cpu()
                entropy = self.conditional_entropy(ts_cpu)
                print("CUDA out of memory handling")

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return entropy
            return math.nan
        except Exception as e:
            print("Conditional_entropy_ ", e)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def joint_entropy(self, tensor):
        try:
            # probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor
            log_probs = torch.log2(probs + 1e-9)  # Adding small value to avoid log(0).
            entropy = -torch.sum(probs * log_probs).item()

            del log_probs
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            return entropy

        except RuntimeError as e:
            print("Joint_entropy RuntimeError ", e)
            if "CUDA out of memory" in str(e):
                print(tensor.shape)
                ts_cpu = tensor.detach().cpu()
                entropy = self.joint_entropy(ts_cpu)
                print("CUDA out of memory handling")

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return entropy

            return math.nan

        except Exception as e:
            print("Joint_entropy ", e)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def renyi_entropy(self, tensor, alpha):
        try:
            # util.endl_time("renyi_entropy start")

            # probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor

            # logit = self.get_logits(torch.log(torch.sum(Categorical(probs=probs).probs ** alpha)))

            # avoid 0
            logit = self.get_logits(
                torch.log(torch.sum((Categorical(probs=probs).probs) ** alpha))
            )

            entropy = 1 / (1 - alpha) * logit

            ################
            # Last Version
            # logit = self.get_logits(torch.log(torch.sum(probs**alpha)))
            # entropy = 1 / (1 - alpha) * logit
            ################

            # entropy = (1 / (1 - alpha) * torch.log(torch.sum(probs**alpha, dim=len(tensor.shape) - 1)).mean()).item()

            # print(torch.sum(probs**alpha, dim=len(tensor.shape) - 1))
            # print(torch.min(torch.sum(probs**alpha, dim=len(tensor.shape) - 1)))
            # print(entropy)
            # input()

            # Convert tensor into a categorical distribution
            # dist = Categorical(probs=tensor)

            # print(1 / (1 - alpha) * logit)

            # entropy = -torch.sum(probs * logit).item()

            # Compute Renyi entropy
            # entropy = 1 / (1 - alpha) * torch.log(torch.sum(dist.probs**alpha))

            # print(torch.sum(dist.probs**alpha))

            # print(entropy)
            # input()

            del probs, logit, tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # util.endl_time("renyi_entropy")

            return entropy.item()

        except RuntimeError as e:
            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("renyi_entropy RuntimeError", e)

            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan
        except Exception as e:
            if "invalid values" in str(e):
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
                entropy = (
                    1
                    / (1 - alpha)
                    * torch.log(
                        torch.sum(probs**alpha, dim=len(tensor.shape) - 1) + 1e-10
                    ).mean()
                ).item()

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                return entropy

            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("renyi_entropy ", e)

            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def tsallis_entropy(self, tensor, q):
        try:
            # util.endl_time("tsallis_entropy start")

            # probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor

            # dist = Categorical(probs=probs)

            #################################
            # Last Version
            # entropy = ((1 - torch.sum(probs ** q)) / (q - 1)).mean().item()

            entropy = (
                ((1 - torch.sum(Categorical(probs=probs).probs ** q)) / (q - 1))
                .mean()
                .item()
            )

            # probs = probs

            # # print(torch.sum(dist.probs))
            # # print(probs)
            # # print(probs**q)
            # # print(torch.sum(probs**q))

            # print(entropy)
            # print(((1 - torch.sum(probs**q)) / (q - 1)).mean().item())
            # input()

            del probs, tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # util.endl_time("tsallis_entropy")

            return entropy
        except RuntimeError as e:
            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("tsallis_entropy RuntimeError", e)

            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

        except Exception as e:
            if "invalid values" in str(e):
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
                entropy = ((1 - torch.sum(probs**q)) / (q - 1)).mean().item()

                del tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                return entropy

            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("tsallis_entropy Exception", e)

            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return math.nan

    def differential_entropy_tensor(self, tensor):
        try:
            # util.endl_time("differential_entropy_tensor start")

            # probs = torch.softmax(-tensor, dim=len(tensor.shape) - 1)
            if cfgs.probs_enable:
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
            else:
                probs = tensor

            # dist = Categorical(probs=probs)

            # pairwise_distance_squared = dist.probs**2
            # pairwise_distance_squared = probs**2

            # # Average squared pairwise distances to get sample variance
            # variance = torch.mean(pairwise_distance_squared)

            # logits = self.get_logits(torch.log(2 * torch.tensor(torch.pi).to(device) * variance))

            # Number of dimensions (n) is just the second dimension size
            n = tensor.size(1)

            ############################################
            # Last Version
            # flattened_distances = probs.flatten()
            ############################################

            flattened_distances = Categorical(probs=probs).probs.flatten()

            # print(flattened_distances)
            # input()

            # Calculate the volume of the space
            volume = (4 / 3) * math.pi * (flattened_distances.max() ** 3)

            # Calculate differential entropy
            entropy = torch.log(volume / n).item()

            # Compute differential entropy for Gaussian
            # entropy = 0.5 * n * (1.0 + torch.log(2 * torch.tensor([3.14159265358979323846]).to(device) * variance))
            # entropy = 0.5 * n * (1.0 + logits)

            del tensor, flattened_distances, volume
            del probs
            del n
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # util.endl_time("differential_entropy_tensor")

            return entropy

        except RuntimeError as e:
            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("differential_entropy RuntimeError", e)
            return math.nan

        except Exception as e:
            if "invalid values" in str(e):
                probs = torch.softmax(tensor, dim=len(tensor.shape) - 1)
                # Number of dimensions (n) is just the second dimension size
                n = tensor.size(1)

                ############################################
                # Last Version
                # flattened_distances = probs.flatten()
                ############################################

                flattened_distances = probs.flatten()

                # print(flattened_distances)
                # input()

                # Calculate the volume of the space
                volume = (4 / 3) * math.pi * (flattened_distances.max() ** 3) + 1e-10

                # Calculate differential entropy
                entropy = torch.log(volume / n).item()

                del tensor, flattened_distances, volume
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                return entropy

            print(
                f"{cfgs.CURRENT_BATCH}_{cfgs.CURRENT_REL}_{cfgs.CURRENT_HIT}_{cfgs.CURRENT_MODEL}_{cfgs.CURRENT_LABEL}"
            )
            print("differential_entropy Exception", e)
            return math.nan

    def differential_entropy(self, tensor):
        try:
            # Calculate pairwise distances
            distances = torch.cdist(tensor, tensor)

            # Flatten the distance matrix into a 1D tensor
            flattened_distances = distances.flatten()

            # Calculate the number of vectors
            n_vectors = flattened_distances.size(0)

            # Calculate the volume of the space
            volume = (4 / 3) * math.pi * (flattened_distances.max() ** 3)

            # Calculate differential entropy
            differential_entropy = torch.log(volume / n_vectors).item()

            del tensor
            del distances
            del flattened_distances
            del n_vectors
            del volume
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            return differential_entropy

        except Exception as e:
            print(e)
            return math.nan

    def diff_ent_cos_dist(self, resource, bandwidth=0.1):
        try:
            # Flatten the distance matrix into a 1D tensor
            flattened_distances = resource.flatten()

            # Calculate the number of vectors
            n_vectors = flattened_distances.size(0)

            # Calculate the volume of the space
            volume = (4 / 3) * math.pi * (flattened_distances.max() ** 3)

            # Calculate differential entropy
            differential_entropy = torch.log(volume / n_vectors).item()

            del flattened_distances
            del n_vectors
            del volume
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            return differential_entropy
        except Exception as e:
            print(e)
            return math.nan

    def diff_entropy_cos_sim_dist(self, resource, bandwidth=0.1):
        try:
            # Flatten the distance matrix into a 1D tensor
            flattened_distances = resource.flatten()

            # Calculate the number of vectors
            n_vectors = flattened_distances.size(0)

            # Calculate the volume of the space
            volume = (4 / 3) * math.pi * (flattened_distances.max() ** 3)

            # Calculate differential entropy
            differential_entropy = torch.log(volume / n_vectors).item()

            del resource
            del flattened_distances
            del n_vectors
            del volume
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            return differential_entropy
        except Exception as e:
            print(e)
            return math.nan

    def calculate_entropies(self):
        # alpha = 0.5
        # q = 1.5
        # rounds = 7

        if "DEFAULT" in cfgs.CALC_MODE:
            if "EMPTY" in cfgs.MODE:
                return (0.0,) * 12

            return (
                round(self.shannon_entropy(self.n_dist), self.rounds),
                # round(self.joint_entropy(self.n_dist), self.rounds),
                round(self.renyi_entropy(self.n_dist, self.alpha), self.rounds),
                round(self.tsallis_entropy(self.n_dist, self.q), self.rounds),
                round(self.differential_entropy_tensor(self.n_dist), self.rounds),
                #
                round(self.differential_entropy_tensor(self.pairwise_sim), self.rounds),
                round(self.shannon_entropy(self.pairwise_sim), self.rounds),
                # round(self.joint_entropy(self.pairwise_sim), self.rounds),
                round(self.renyi_entropy(self.pairwise_sim, self.alpha), self.rounds),
                round(self.tsallis_entropy(self.pairwise_sim, self.q), self.rounds),
                #
                round(
                    self.differential_entropy_tensor(self.mahalanobis_dist), self.rounds
                ),
                round(self.shannon_entropy(self.mahalanobis_dist), self.rounds),
                # round(self.joint_entropy(self.mahalanobis_dist), self.rounds),
                round(
                    self.renyi_entropy(self.mahalanobis_dist, self.alpha), self.rounds
                ),
                round(self.tsallis_entropy(self.mahalanobis_dist, self.q), self.rounds),
                #
                self.n_dist_p,
                self.pairwise_sim_p,
                self.mahalanobis_dist_p,
            )

        else:
            return (
                self.n_dist,
                self.pairwise_sim,
                self.mahalanobis_dist,
                #
                self.n_dist_p,
                self.pairwise_sim_p,
                self.mahalanobis_dist_p,
            )

    def calculate_entropies_no_maha(self):
        if "DEFAULT" in cfgs.CALC_MODE:
            if "EMPTY" in cfgs.MODE:
                return (0.0,) * 12

            return (
                round(self.shannon_entropy(self.n_dist), self.rounds),
                # round(self.joint_entropy(self.n_dist), self.rounds),
                round(self.renyi_entropy(self.n_dist, self.alpha), self.rounds),
                # round(self.tsallis_entropy(self.n_dist, self.q), self.rounds),
                0.0,
                # round(self.differential_entropy_tensor(self.n_dist), self.rounds),
                0.0,
                #
                # round(self.differential_entropy_tensor(self.pairwise_sim), self.rounds),
                0.0,
                round(self.shannon_entropy(self.pairwise_sim), self.rounds),
                # round(self.joint_entropy(self.pairwise_sim), self.rounds),
                round(self.renyi_entropy(self.pairwise_sim, self.alpha), self.rounds),
                # round(self.tsallis_entropy(self.pairwise_sim, self.q), self.rounds),
                0.0,
                #
                self.n_dist_p,
                self.pairwise_sim_p,
            )

        else:
            return (
                self.n_dist,
                self.pairwise_sim,
                #
                self.n_dist_p,
                self.pairwise_sim_p,
            )

    def calculate_entropies_renyi_and_tsallis(self):
        alpha = 0.15
        q = 1.95
        rounds = 7

        return (
            round(self.renyi_entropy(self.tensor, alpha), rounds),
            round(self.tsallis_entropy(self.tensor, q), rounds),
            round(self.renyi_entropy(self.pairwise_sim, alpha), rounds),
            round(self.tsallis_entropy(self.pairwise_sim, q), rounds),
        )

    def calcTest(self):
        # round(self.shannon_entropy(self.tensor), 7),
        return round(self.shannon_entropy(self.pairwise_sim_1dim), 7)
