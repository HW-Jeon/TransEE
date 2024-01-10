import math

import torch

import config_TransEE as cfgs
import utilities as util


class PDF_Fitters:
    def kde(self, _x, _sample, bandwidth):
        """
        Compute the KDE using the Gaussian kernel for a given sample and bandwidth.
        x: Tensor where to evaluate the KDE.
        sample: Tensor of the pairwise distances.
        bandwidth: Scalar for the bandwidth of the Gaussian kernel.
        """
        try:
            assert bandwidth > 0, "Bandwidth must be positive."

            # util.check_CUDA_MEM("IN kde init")

            # Make sure x is a column vector and sample is a row vector
            with torch.no_grad():
                x = _x.unsqueeze(1)
                sample = _sample.unsqueeze(0)

                # util.check_CUDA_MEM("IN kde init 2")

                # Compute the KDE

                exponent = -0.5 * ((x - sample) / bandwidth) ** 2

                # util.check_CUDA_MEM("IN kde exponent")
                kernel_vals = torch.exp(exponent) / (torch.sqrt(torch.tensor(2 * torch.pi)) * bandwidth)

                # util.check_CUDA_MEM("IN kde kernel_vals")
                pdf_vals = torch.mean(kernel_vals, dim=1)

            # util.check_CUDA_MEM("IN kde pdf_vals")

            # print("_x size: ", _x.element_size() * _x.nelement() / (1024**2))
            # print("_sample size: ", _sample.element_size() * _sample.nelement() / (1024**2))
            # print("sample size: ", sample.element_size() * sample.nelement() / (1024**2))
            # print("bandwidth size: ", bandwidth.element_size() * bandwidth.nelement() / (1024**2))
            # print("exponent size: ", exponent.element_size() * exponent.nelement() / (1024**2))
            # print("kernel_vals size: ", kernel_vals.element_size() * kernel_vals.nelement() / (1024**2))

            del _x, _sample, x, sample, bandwidth, exponent, kernel_vals
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # print(pdf_vals)
            # print("result size: ", pdf_vals.element_size() * pdf_vals.nelement() / (1024**2))

            # util.check_CUDA_MEM("IN kde release")

            # input()

            return pdf_vals.clone()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # print("kde RuntimeError, CUDA out of memory")
                util.check_CUDA_MEM("kde RuntimeError, CUDA out of memory")
                x = _x.detach().cpu()
                ts_cpu = _sample.detach().cpu()
                bandwidths = bandwidth.detach().cpu()
                distance = self.kde(x, ts_cpu, bandwidths)

                del _x, _sample, bandwidth, x, ts_cpu, bandwidths
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                print(distance.shape)
                return distance.to(cfgs.devices)
            else:
                print("pairwise_mahalanobis_distance RuntimeError", e)
            return math.nan

    def silverman_bandwidth(self, sample):
        """
        Estimate the bandwidth using Silverman's rule of thumb.
        sample: Tensor of the pairwise distances.
        """
        n = sample.shape[0]
        std_dev = torch.std(sample)

        result = 1.06 * std_dev * n ** (-1 / 5)

        del sample, n, std_dev
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return result

    def compute_pdf_for_each_value(self, sample):
        try:
            """
            Compute the probability distribution of a given tensor using KDE for each value.
            sample: Tensor of the pairwise distances.
            """
            # Estimate bandwidth
            bandwidth = self.silverman_bandwidth(sample)

            # Compute the KDE for each value in the sample

            pdf_values_list = []

            for i in range(0, len(sample), cfgs.pd_batch_size):
                batch = sample[i : i + cfgs.pd_batch_size]
                with torch.no_grad():
                    pdf_batch = self.kde(batch, sample, bandwidth)
                    pdf_values_list.append(pdf_batch)

                del batch, pdf_batch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            result = torch.cat(pdf_values_list)

            del sample, bandwidth, pdf_values_list
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # print(type(pdf_vals))

            return result.clone()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("compute_pdf_for_each_value RuntimeError, CUDA out of memory")
                ts_cpu = sample.detach().cpu()

                print(ts_cpu.shape)
                distance = self.compute_pdf_for_each_value(ts_cpu).to(cfgs.devices)
                print("CUDA out of memory handling")
                del sample
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                print(distance.shape)
                return distance.clone()
            else:
                print("pairwise_mahalanobis_distance RuntimeError", e)
            return math.nan
