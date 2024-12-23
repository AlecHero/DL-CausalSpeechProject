
'''
@Filename    :Loss.py
@Time        :2020/07/09 22:11:13
@Author      :Kai Li
@Version     :1.0
'''

import torch
from itertools import permutations

# https://github.com/JusperLee/Conv-TasNet/blob/master/Conv-TasNet_lightning/train.py

def sisdr_to_sdr(sisdr, scaling_factor=1.0):
    """
    Estimate SDR from SI-SDR given an assumed scaling factor.

    Parameters:
    sisdr (float): The SI-SDR value in dB.
    scaling_factor (float): The ratio of the estimated signal's energy scaling
                            relative to the target. If 1.0, we assume they are 
                            perfectly matched in scale.

    Returns:
    float: An estimated SDR in dB.
    """
    # If scaling_factor = 1, SI-SDR ~ SDR
    # For a non-unity scaling_factor, a theoretical adjustment could be:
    # SDR ≈ SI-SDR + 20 * log10(scaling_factor)
    # But this requires knowing the scaling_factor beforehand.
    
    import math
    if scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive.")

    # Apply the approximate correction for scaling if known
    correction = 20 * math.log10(scaling_factor) if scaling_factor != 1.0 else 0
    sdr = sisdr + correction
    return sdr

# Example usage:
# If we assume no scaling difference:

# If we somehow knew the estimated signal was scaled by a factor of 2:
# estimated_sdr = sisdr_to_sdr(10.0, scaling_factor=2.0)
# print("Estimated SDR with scaling:", estimated_sdr, "dB")


class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def snr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor 
        s: reference signal, N x S tensor
        Return:
        snr: N tensor
        """
        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate snr, {} vs {}".format(
                    x.shape, s.shape))

        noise = x - s
        return 20 * torch.log10(eps + l2norm(s) / (l2norm(noise) + eps))

    def compute_loss(self, ests, refs):

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = ests[0].size(0)
        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(len(ests)))])
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        # si-snr
        return -torch.sum(max_perutt) / N

    def sisnr_intermediate(self, student_feats, teacher_feats, eps=1e-8):
        """
        SI-SNR loss for intermediate feature alignment.
        Arguments:
        student_feats: List of tensors representing student intermediate features.
        teacher_feats: List of tensors representing teacher intermediate features.
        Return:
        Total SI-SNR loss across all intermediate layers.
        """
        if len(student_feats) != len(teacher_feats):
            raise RuntimeError(
                "Mismatch in number of intermediate features: {} vs {}".format(
                    len(student_feats), len(teacher_feats)))
        
        loss = 0.0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            if s_feat.shape != t_feat.shape:
                raise RuntimeError(
                    "Feature dimension mismatch: {} vs {}".format(s_feat.shape, t_feat.shape))
            # Apply SI-SNR to feature maps
            loss += -torch.mean(self.sisnr(s_feat, t_feat, eps))
        return loss
    
class Accuracy():
    def __init__(self):
        super(Accuracy, self).__init__()
    
    # R^2 / R squared
    def r2(self, y_pred, y_true):
        ssr = torch.sum((y_true - y_pred) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ssr / sst
    
    # MSE / mean squared error
    @staticmethod  
    def mse(y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)
    
    # RMSE / root mean square error 
    def rmse(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

    # MAE / mean absolute error
    def mae(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))    

    
if __name__ == "__main__":
    ests = torch.randn(4,320)
    egs = torch.randn(4,320)
    loss = Loss()
    print(loss.compute_loss(ests, egs))
    print(loss.sisnr(ests, egs))
    print([sisdr_to_sdr(est, scaling_factor=1.2) for est in loss.sisnr(ests, egs)])