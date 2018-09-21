# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import scipy.io
from functions import globals
from functions.math_utils import azim_proj

def load_locations(withProjection=True):
    print('Electrode locations is loading...')
    if withProjection:
        locs = scipy.io.loadmat(globals.file_eeg_locs)
        locs_3d = locs['locs3d']
        locs_2d = []

        # 2D Projection
        for e in locs_3d:
            locs_2d.append(azim_proj(e))

        # Shift to zero
        locs_2d = np.asarray(locs_2d)
        min_0 = locs_2d[:, 0].min()
        min_1 = locs_2d[:, 1].min()

        locs_2d = locs_2d - (min_0, min_1)

    else:
        locs = scipy.io.loadmat(globals.file_eeg_locs2d)
        locs_2d = locs['locs2d']
    return locs_2d


def check_feat_dir(windowLength, overlap):
    # is dir exist
    dir = globals.file_feat_dataset + str(windowLength) + 's_' + str(overlap) + 'o'
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def check_img_dir(windowLength, overlap, genType, pixelCount=8, genMethod='RBF', edgeless=True):
    # is dir exist
    dir = globals.file_img_dataset + str(windowLength) + 's_' + str(overlap) + 'o_' + str(genType)
    if genType == 'advanced':
        dir = dir + '_' + str(pixelCount) + 'p_' + genMethod
        if edgeless:
            dir = dir + '_edgeless'
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def check_label_dir(windowLength, overlap, preictal_len, postictal_len, min_preictal_len, min_interictal_len,
                    excluded_len, sph_len):
    # is dir exist
    dir = globals.file_lbl_dataset + str(windowLength) + 's_' + str(overlap) + 'o_' + str(preictal_len) + 'pre_' + \
          str(postictal_len) + 'post'
    if min_preictal_len:
        dir += '_' + str(min_preictal_len) + 'min_pre'
    if min_interictal_len:
        dir += '_' + str(min_interictal_len) + 'min_int'
    if excluded_len:
        dir += '_' + str(excluded_len) + 'excluded'
    if sph_len:
        dir += '_' + str(sph_len) + 'sph'
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir