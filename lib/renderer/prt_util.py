
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

import os
import trimesh
import numpy as np
import math
from scipy.special import sph_harm
import argparse
from tqdm import tqdm
from trimesh.util import bounds_tree


def factratio(N, D):
    if N >= D:
        prod = 1.0
        for i in range(D + 1, N + 1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N + 1, D + 1):
            prod *= i
        return 1.0 / prod


def KVal(M, L):
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))


def AssociatedLegendre(M, L, x):
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)

    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M + 1):
            pmm = -pmm * fact * somx2
            fact = fact + 2

    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M + 1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M + 2, L + 1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll


def SphericalHarmonic(M, L, theta, phi):
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * np.cos(
            M * phi) * AssociatedLegendre(M, L, np.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * np.sin(
            -M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))


def save_obj(mesh_path, verts):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()


def sampleSphericalDirections(n):
    xv = np.random.rand(n, n)
    yv = np.random.rand(n, n)
    theta = np.arccos(1 - 2 * xv)
    phi = 2.0 * math.pi * yv

    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    vx = -np.sin(theta) * np.cos(phi)
    vy = -np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    return np.stack([vx, vy, vz], 1), phi, theta


def getSHCoeffs(order, phi, theta):
    shs = []
    for n in range(0, order + 1):
        for m in range(-n, n + 1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)

    return np.stack(shs, 1)


def computePRT(mesh_path, scale, n, order):

    prt_dir = os.path.join(os.path.dirname(mesh_path), "prt")
    bounce_path = os.path.join(prt_dir, "bounce.npy")
    face_path = os.path.join(prt_dir, "face.npy")

    os.makedirs(prt_dir, exist_ok=True)

    PRT = None
    F = None

    if os.path.exists(bounce_path) and os.path.exists(face_path):

        PRT = np.load(bounce_path)
        F = np.load(face_path)

    else:

        mesh = trimesh.load(mesh_path,
                            skip_materials=True,
                            process=False,
                            maintain_order=True)
        mesh.vertices *= scale

        vectors_orig, phi, theta = sampleSphericalDirections(n)
        SH_orig = getSHCoeffs(order, phi, theta)

        w = 4.0 * math.pi / (n * n)

        origins = mesh.vertices
        normals = mesh.vertex_normals
        n_v = origins.shape[0]

        origins = np.repeat(origins[:, None], n, axis=1).reshape(-1, 3)
        normals = np.repeat(normals[:, None], n, axis=1).reshape(-1, 3)
        PRT_all = None
        for i in range(n):
            SH = np.repeat(SH_orig[None, (i * n):((i + 1) * n)], n_v,
                           axis=0).reshape(-1, SH_orig.shape[1])
            vectors = np.repeat(vectors_orig[None, (i * n):((i + 1) * n)],
                                n_v,
                                axis=0).reshape(-1, 3)

            dots = (vectors * normals).sum(1)
            front = (dots > 0.0)

            delta = 1e-3 * min(mesh.bounding_box.extents)

            hits = mesh.ray.intersects_any(origins + delta * normals, vectors)
            nohits = np.logical_and(front, np.logical_not(hits))

            PRT = (nohits.astype(np.float) * dots)[:, None] * SH

            if PRT_all is not None:
                PRT_all += (PRT.reshape(-1, n, SH.shape[1]).sum(1))
            else:
                PRT_all = (PRT.reshape(-1, n, SH.shape[1]).sum(1))

        PRT = w * PRT_all
        F = mesh.faces

        np.save(bounce_path, PRT)
        np.save(face_path, F)

    # NOTE: trimesh sometimes break the original vertex order, but topology will not change.
    # when loading PRT in other program, use the triangle list from trimesh.

    return PRT, F


def testPRT(obj_path, n=40):

    os.makedirs(os.path.join(os.path.dirname(obj_path),
                             f'../bounce/{os.path.basename(obj_path)[:-4]}'),
                exist_ok=True)

    PRT, F = computePRT(obj_path, n, 2)
    np.savetxt(
        os.path.join(os.path.dirname(obj_path),
                     f'../bounce/{os.path.basename(obj_path)[:-4]}',
                     'bounce.npy'), PRT)
    np.save(
        os.path.join(os.path.dirname(obj_path),
                     f'../bounce/{os.path.basename(obj_path)[:-4]}',
                     'face.npy'), F)
