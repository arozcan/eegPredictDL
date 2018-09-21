# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pickle
import scipy.io
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold
from functions.io_utils import check_img_dir, check_label_dir, check_feat_dir
from functions import globals
from functions.model import create_model
from functions.train import train_model
from functions.math_utils import normal_prob_signal
from functions.plot_data import plot_subject_prediction, plot_subject_scores


# deterministic random
if globals.random_state != None:
    np.random.seed(globals.random_state)


def calc_label_weights(labels, interictalParts, preictalParts, sigma=2, defaultPreictalLen=0, stepLen=2):
    defaultPreictalLen = np.int(defaultPreictalLen * 60 / stepLen)
    weights = np.zeros(len(labels))
    for p in preictalParts:
        if defaultPreictalLen:
            signalLen = p[-1] - p[0] + 1
            weights[p] = normal_prob_signal(p[-1]-defaultPreictalLen+1, p[-1], sigma=sigma)[-signalLen:]
        else:
            weights[p]=normal_prob_signal(p[0], p[-1],sigma=sigma)
    for i in interictalParts:
        weights[i] = 1
    return weights



def load_data(featureParams, timingParams, subject):

    # check label dataset directory
    lbl_dir = check_label_dir(featureParams.windowLength, featureParams.overlap, timingParams.preictalLen,
                              timingParams.postictalLen, timingParams.minPreictalLen, timingParams.minInterictalLen,
                              timingParams.excludedLen, timingParams.sphLen)

    if featureParams.genType == 'none':
        # check feature dataset directory
        feat_dir = check_feat_dir(featureParams.windowLength, featureParams.overlap)
    else:
        # check feature dataset directory
        img_dir = check_img_dir(featureParams.windowLength, featureParams.overlap, featureParams.genType,
                                featureParams.pixelCount[0], featureParams.genMethod, featureParams.edgeless)

    subject_file_name = lbl_dir + '/' + 'subject_' + str(subject) + '.mat'

    if os.path.isfile(subject_file_name):
        # open subject label data
        pickle_in = open(subject_file_name, "rb")
        subject_ = pickle.load(pickle_in)
        pickle_in.close()

        # calculate label weigts
        weights = calc_label_weights(subject_.all_labels, subject_.interictal_parts, subject_.preictal_parts, sigma=2,
                                     defaultPreictalLen=timingParams.preictalLen,
                                     stepLen=(featureParams.windowLength*(1 - featureParams.overlap)))

        data_all = []
        for record in subject_.records:
            if featureParams.genType == 'none':
                feat_file_name = feat_dir + '/' + 'record_' + str(record[5]) + '.mat'
                features = scipy.io.loadmat(feat_file_name)['features']
                features = np.swapaxes(features, 1, 2)
                features = features[:,featureParams.featureTypes]
                data_all = features if data_all == [] else np.vstack((data_all, features))
            else:
                img_file_name = img_dir + '/' + 'img_record_' + str(record[5]) + '.mat'
                images_timewin = scipy.io.loadmat(img_file_name)['images_timewin']
                images_timewin = images_timewin[:, featureParams.featureTypes]
                data_all = images_timewin if data_all == [] else np.vstack((data_all,images_timewin))

        if len(data_all) != len(subject_.all_labels):
            exit('Dataset Length Error!')
        return data_all, subject_.all_labels, weights, subject_.interictal_parts, subject_.preictal_parts, subject_.excluded_parts
    else:
        exit('Dataset Load Error!')

def organize_dataset(dataset, interictal_parts, preictal_parts, excluded_parts, foldCount=5, seqWinCount=5,
                     kFoldShuffle=True, underSample = True, evalExclued = False):

    # remove last seqWinCount sample from all parts
    for p in range(len(interictal_parts)):
        interictal_parts[p] = interictal_parts[p][:-seqWinCount]
    for p in range(len(preictal_parts)):
        preictal_parts[p] = preictal_parts[p][:-seqWinCount]
    for p in range(len(excluded_parts)):
        excluded_parts[p] = excluded_parts[p][:-seqWinCount]

    # concatenate interictal parts
    interictal_data = []
    for part in interictal_parts:
        interictal_data = part if interictal_data == [] else np.hstack((interictal_data, part))
    # split interictal data to preictal part count
    interictal_split = np.array_split(interictal_data, len(preictal_parts))

    # concatenate excluded parts
    if evalExclued:
        excluded_data = []
        for part in excluded_parts:
            excluded_data = part if excluded_data == [] else np.hstack((excluded_data, part))
        # split excluded data to preictal part count
        excluded_split = np.array_split(excluded_data, len(preictal_parts))
        dataset.labels[dataset.labels == globals.labelTypes["excluded"]] = globals.labelTypes["interictal"]

    interictal_groups = []
    interictal_data = []
    for part, p in zip(interictal_split, range(len(interictal_split))):
        interictal_groups = np.full((len(part)), p) if interictal_groups == [] else np.hstack((interictal_groups, np.full((len(part)), p)))
        interictal_data = part if interictal_data == [] else np.hstack((interictal_data, part))

    preictal_groups = []
    preictal_data = []
    for part, p in zip(preictal_parts, range(len(preictal_parts))):
        preictal_groups = np.full((len(part)), p) if preictal_groups == [] else np.hstack((preictal_groups, np.full((len(part)), p)))
        preictal_data = part if preictal_data == [] else np.hstack((preictal_data, part))

    # leave one group out for testing and others for training
    logo = LeaveOneGroupOut()
    kf = KFold(n_splits=foldCount, shuffle=kFoldShuffle, random_state=globals.random_state)

    interictal_fold = []
    for i, (train_index, test_index) in zip(range(len(interictal_split)), logo.split(interictal_data, groups=interictal_groups)):
        interictal_train, interictal_test, train_groups = interictal_data[train_index], interictal_data[test_index], \
                                                          interictal_groups[train_index]
        # merge excluded part with interictal test
        if evalExclued:
            interictal_test = np.hstack((interictal_test, excluded_split[i]))

        # K fold cross-validation
        i_fold = [[[], [], interictal_test] for i in range(foldCount)]
        fold_order = range(foldCount)
        u, u_index = np.unique(train_groups, return_index=True)
        u_index = np.append(u_index, len(train_groups))
        u_index = [range(u_index[i], u_index[i + 1]) for i in range(len(u))]
        for index in u_index:
            np.random.shuffle(fold_order)
            for [train_index, valid_index], f in zip(kf.split(interictal_train[index]), fold_order):
                int_train, int_valid = interictal_train[index][train_index], interictal_train[index][valid_index]
                i_fold[f][0] = int_train if i_fold[f][0] == [] else np.hstack((i_fold[f][0], int_train))
                i_fold[f][1] = int_valid if i_fold[f][1] == [] else np.hstack((i_fold[f][1], int_valid))
        interictal_fold.append(i_fold)

    # change interictal fold order
    np.random.shuffle(interictal_fold)
    for i in range(len(interictal_fold)):
        np.random.shuffle(interictal_fold[i])

    preictal_fold = []
    for train_index, test_index in logo.split(preictal_data, groups=preictal_groups):
        preictal_train, preictal_test, train_groups = preictal_data[train_index], preictal_data[test_index], \
                                                      preictal_groups[train_index]
        # K fold cross-validation
        p_fold = [[[], [], preictal_test] for i in range(foldCount)]
        fold_order = range(foldCount)
        u, u_index = np.unique(train_groups, return_index=True)
        u_index = np.append(u_index, len(train_groups))
        u_index = [range(u_index[i], u_index[i + 1]) for i in range(len(u))]
        for index in u_index:
            np.random.shuffle(fold_order)
            for [train_index, valid_index], f in zip(kf.split(preictal_train[index]), fold_order):
                pre_train, pre_valid = preictal_train[index][train_index], preictal_train[index][valid_index]
                p_fold[f][0] = pre_train if p_fold[f][0] == [] else np.hstack((p_fold[f][0], pre_train))
                p_fold[f][1] = pre_valid if p_fold[f][1] == [] else np.hstack((p_fold[f][1], pre_valid))
        preictal_fold.append(p_fold)

    fold_pairs = []
    for i_fold, p_fold in zip(interictal_fold, preictal_fold):
        k_fold = []
        for i, p in zip(i_fold, p_fold):
            if underSample:
                # shuffle interictal train data
                np.random.shuffle(i[0])
                # undersample interictal train data
                i[0] = i[0][:len(p[0])]
                # shuffle interictal valid data
                np.random.shuffle(i[1])
                # undersample interictal valid data
                i[1] = i[1][:len(p[1])]
            # merge interictal and preictal folds
            train = np.hstack((i[0], p[0]))
            valid = np.hstack((i[1], p[1]))
            test = np.hstack((i[2], p[2]))

            # shuffle interictal train data
            np.random.shuffle(train)

            k_fold.append([train, valid, test])
        fold_pairs.append(k_fold)

    if underSample:
        ratio = [1, 1]
    else:
        ratio = [len(preictal_data)/len(interictal_data), 1]

    return fold_pairs, ratio


def predict_data(featureParams, timingParams, trainDataParams, trainParams, modelParams):

    dataset = globals.Dataset()

    # Load data and labels
    dataset.data, dataset.labels, dataset.weights, interictal_parts, preictal_parts, excluded_parts\
        = load_data(featureParams, timingParams, trainDataParams.subject)

    # Organize dataset as train, valid and test sets
    dataset.fold_pairs, trainDataParams.dataRatio = organize_dataset(dataset,
                                                                     interictal_parts, preictal_parts, excluded_parts,
                                                                     foldCount=trainDataParams.foldCount,
                                                                     seqWinCount=trainDataParams.seqWinCount,
                                                                     kFoldShuffle=trainDataParams.kFoldShuffle,
                                                                     underSample=trainDataParams.underSample,
                                                                     evalExclued=trainDataParams.evalExclued )

    # Create learning model
    model = create_model(modelParams, featureParams, trainParams, trainDataParams)

    # Train model
    trainResult = train_model(model, dataset)

    plot_subject_prediction(trainResult.testParams, dataset.fold_pairs, dataset.labels)

    plot_subject_scores(trainResult.testParams, dataset.fold_pairs, dataset.labels, featureParams)


