#!/usr/bin/env python3

from .worker import main
from . import utils
from . import cProjector
from . import differencial
import numpy


def diff_calc_scale(A, subsets):
    # けいさんめも｡
    # 投影空間は普通に足して1/2
    # 画像空間は上0.5と下0.5のpartial_backwardを足す
    NoI, _ = A.get_image_shape()
    NoA, NoD = A.get_projector_shape()
    absorpA = projector.Projector(NoI, NoA, NoD+1)
    a_ip = utils.empty_proj(A)
    a_pj_subset = [utils.empty_img(A) for i in range(len(subsets))]

    # calc a_i+
    img = utils.empty_img(absorpA)
    proj = utils.empty_proj(absorpA)
    img[:, :] = 1
    absorpA.forward(img, proj)
    a_ip = 0.5 * (proj[:, :-1] + proj[:, 1:])

    # calc a_+j
    proj_u = utils.zero_proj(absorpA)
    proj_d = utils.zero_proj(absorpA)
    proj_u[:, :] = 0.5
    proj_d[:, :] = 0.5
    img_u = utils.empty_img(absorpA)
    img_d = utils.empty_img(absorpA)
    for i in range(len(subsets)):
        absorpA.partial_backward(proj_u, img_u, subsets[i])
        absorpA.partial_backward(proj_d, img_d, subsets[i])
        a_pj_subset[i] = img_u + img_d
        a_pj_subset[i][a_pj_subset[i]==0] = 1  # avoid zero-division

    return a_ip, a_pj_subset


def calc_scale(A, subsets):
    if isinstance(A, differencial.Projector):
        print("use diff_calc_scale")
        raise NotImplementedError("differencial version is unsupported")
        return diff_calc_scale(A, subsets)

    if not isinstance(A, cProjector.Projector):
        raise NotImplementedError("unknown projector matrix")

    # 吸収版｡
    a_ip = utils.zero_proj(A)
    img = utils.zero_img(A)
    img[:, :] = 1
    A.forward(img, a_ip)
    a_ip[a_ip == 0] = 1

    a_pj_subset = [utils.zero_img(A) for i in range(len(subsets))]
    proj = utils.zero_proj(A)
    proj[:, :] = 1
    for i in range(len(subsets)):
        A.partial_backward(proj, a_pj_subset[i], subsets[i])
        a_pj_subset[i][a_pj_subset[i]==0] = 1  # avoid zero-division

    return a_ip, a_pj_subset


def make_subset(NoA, n_subset, shuffled=False):
    mod = NoA % n_subset
    unit_len = NoA // n_subset
    w = (NoA+mod) // n_subset
    h = n_subset
    index1d = numpy.array([j*h + i for i in range(h) for j in range(w) if j*h + i < NoA])
    h, t = 0, unit_len
    index2d = []
    for i in range(n_subset):
        if i < mod:
            t += 1
        index2d.append(index1d[h:t])
        h = t
        t += unit_len
    if shuffled:
        numpy.random.shuffle(index2d)
    return index2d

def os_sart_mainloop(A, data, recon, alpha, a_ip, a_pj_subset, subsets, i, img, proj):
    i_subset = i % len(subsets)
    cur_subset = subsets[i_subset]
    A.partial_forward(recon, proj, cur_subset)
    proj -= data
    proj /= -a_ip
    A.partial_backward(proj, img, cur_subset)
    img /= a_pj_subset[i_subset]
    recon += alpha * img
