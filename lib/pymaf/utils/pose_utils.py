"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt(((S1_hat - S2)**2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, S1_hat


# https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
def axis_angle_add(theta, roll_axis, alpha):
    """Composition of two axis-angle rotations (PyTorch version)
    Args:
        theta: N x 3
        roll_axis: N x 3
        alph: N x 1
    Returns:
        equivalent axis-angle of the composition
    """
    l2norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l2norm, -1)

    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    b_cos = torch.cos(angle).cpu()
    b_sin = torch.sin(angle).cpu()

    a_cos = torch.cos(alpha)
    a_sin = torch.sin(alpha)

    dot_mm = torch.sum(normalized * roll_axis, dim=1, keepdim=True)
    cross_mm = torch.zeros_like(normalized)
    cross_mm[:,
             0] = roll_axis[:,
                            1] * normalized[:,
                                            2] - roll_axis[:,
                                                           2] * normalized[:,
                                                                           1]
    cross_mm[:,
             1] = roll_axis[:,
                            2] * normalized[:,
                                            0] - roll_axis[:,
                                                           0] * normalized[:,
                                                                           2]
    cross_mm[:,
             2] = roll_axis[:,
                            0] * normalized[:,
                                            1] - roll_axis[:,
                                                           1] * normalized[:,
                                                                           0]

    c_cos = a_cos * b_cos - a_sin * b_sin * dot_mm
    c_sin_n = a_sin * b_cos * roll_axis + a_cos * b_sin * normalized + a_sin * b_sin * cross_mm

    c_angle = 2 * torch.acos(c_cos)
    c_sin = torch.sin(c_angle * 0.5)
    c_n = (c_angle / c_sin) * c_sin_n

    return c_n


def axis_angle_add_np(theta, roll_axis, alpha):
    """Composition of two axis-angle rotations (NumPy version)
    Args:
        theta: N x 3
        roll_axis: N x 3
        alph: N x 1
    Returns:
        equivalent axis-angle of the composition
    """

    angle = np.linalg.norm(theta + 1e-8, ord=2, axis=1, keepdims=True)
    normalized = np.divide(theta, angle)
    angle = angle * 0.5

    b_cos = np.cos(angle)
    b_sin = np.sin(angle)
    a_cos = np.cos(alpha)
    a_sin = np.sin(alpha)

    dot_mm = np.sum(normalized * roll_axis, axis=1, keepdims=True)
    cross_mm = np.zeros_like(normalized)
    cross_mm[:,
             0] = roll_axis[:,
                            1] * normalized[:,
                                            2] - roll_axis[:,
                                                           2] * normalized[:,
                                                                           1]
    cross_mm[:,
             1] = roll_axis[:,
                            2] * normalized[:,
                                            0] - roll_axis[:,
                                                           0] * normalized[:,
                                                                           2]
    cross_mm[:,
             2] = roll_axis[:,
                            0] * normalized[:,
                                            1] - roll_axis[:,
                                                           1] * normalized[:,
                                                                           0]

    c_cos = a_cos * b_cos - a_sin * b_sin * dot_mm
    c_sin_n = a_sin * b_cos * roll_axis + a_cos * b_sin * normalized + a_sin * b_sin * cross_mm
    c_angle = 2 * np.arccos(c_cos)
    c_sin = np.sin(c_angle * 0.5)
    c_n = (c_angle / c_sin) * c_sin_n

    return c_n
