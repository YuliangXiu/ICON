# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math

import numpy as np
import torch
from loguru import logger
from torch import nn as nn


class AleatoricLoss(nn.Module):
    def __init__(self):
        super(AleatoricLoss, self).__init__()

    def forward(self, pred, gt):
        n_dims = int(int(pred.shape[1]) // 2)
        mu = pred[:, :n_dims]
        sigma = pred[:, n_dims:]

        se = torch.pow((gt - mu), 2)
        inv_std = torch.exp(-sigma)
        mse = torch.mean(inv_std * se)
        reg = torch.mean(sigma)
        return 0.5 * (mse + reg)


class MSE_VAR(nn.Module):
    def __init__(self, var_weight=1.0):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, pred, gt):
        n_dims = int(int(pred.shape[1]) // 2)
        mean = pred[:, :n_dims]
        var = pred[:, n_dims:]

        var = self.var_weight * var

        loss1 = torch.mul(torch.exp(-var), (mean - gt)**2)
        loss2 = var
        loss = .5 * (loss1 + loss2)
        return loss.mean()


class MultivariateGaussianNegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(MultivariateGaussianNegativeLogLikelihood, self).__init__()

    def forward(self, pred, gt):
        n_dims = int(int(pred.shape[1]) / 2)
        mu = pred[:, :n_dims]
        logsigma = pred[:, n_dims:]
        # logsigma = torch.pow(pred[:, n_dims:], 2)

        mse = -0.5 * torch.sum(torch.square((gt - mu) / torch.exp(logsigma)),
                               dim=1)
        sigma_trace = -torch.sum(logsigma, dim=1)
        log2pi = -0.5 * n_dims * np.log(2 * np.pi)

        log_likelihood = mse + sigma_trace + log2pi
        total_loss = torch.mean(-log_likelihood)

        # print(f'\nMSE: {mse.mean().item():.2f}'
        #       f' Sigma: {sigma_trace.mean().item():.2f}'
        #       f' log2pi:{log2pi.mean():.2f}'
        #       f' Total:{total_loss.item():.2f}')
        # logger.debug(f'\nMSE: {mse.mean().item():.2f}'
        #              f' Sigma: {sigma_trace.mean().item():.2f}'
        #              f' log2pi:{log2pi.mean():.2f}'
        #              f' Total:{total_loss.item():.2f}')

        return total_loss


class LaplacianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """
    def __init__(self, size_average=True, reduce=True, evaluate=False):
        super(LaplacianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate

    def laplacian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py

        """

        n_dims = int(int(mu_si.shape[1]) / 2)

        mu, si = mu_si[:, :n_dims], mu_si[:, n_dims:]
        # norm = xx - mu
        norm = 1 - mu / xx  # Relative

        term_a = torch.abs(norm) * torch.exp(-si)
        term_b = si
        norm_bi = (np.mean(np.abs(norm.cpu().detach().numpy())),
                   np.mean(torch.exp(si).cpu().detach().numpy()))

        if self.evaluate:
            return norm_bi
        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.laplacian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            logger.debug(f'Mean: {mean_values.item()}')
            return mean_values
        return torch.sum(values)


class GaussianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """
    def __init__(self, device, size_average=True, reduce=True, evaluate=False):
        super(GaussianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate
        self.device = device

    def gaussian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """
        n_dims = int(int(mu_si.shape[1]) / 2)

        mu, si = mu_si[:, :n_dims], mu_si[:, n_dims:]

        min_si = torch.ones(si.size()).cuda(self.device) * 0.1
        si = torch.max(min_si, si)
        norm = xx - mu
        term_a = (norm / si)**2 / 2
        term_b = torch.log(si * math.sqrt(2 * math.pi))

        norm_si = (np.mean(np.abs(norm.cpu().detach().numpy())),
                   np.mean(si.cpu().detach().numpy()))

        if self.evaluate:
            return norm_si

        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.gaussian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


# Losses from https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-
# Mixture-Density-Network/blob/master/src/mix_den_model.py


def mean_log_Gaussian_like(y_true, parameters, c, m):
    """Mean Log Gaussian Likelihood distribution
    y_truth: ground truth 3d pose
    parameters: output of hypotheses generator, which conclude the mean, variance and mixture coeffcient of the mixture model
    c: dimension of 3d pose
    m: number of kernels
    """
    components = tf.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    sigma = tf.clip_by_value(sigma, 1e-15, 1e15)
    alpha = components[:, c + 1, :]
    alpha = tf.clip_by_value(alpha, 1e-8, 1.)

    exponent = tf.log(alpha) - 0.5 * c * tf.log(2 * np.pi) \
               - c * tf.log(sigma) \
               - tf.reduce_sum((tf.expand_dims(y_true, 2) - mu) ** 2, axis=1) / (2.0 * (sigma) ** 2.0)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = -tf.reduce_mean(log_gauss)
    return res


def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    return tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis,
                                keep_dims=True)) + x_max
