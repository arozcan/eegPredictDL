# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pickle
import numpy as np
import pyedflib
import scipy.io

from functions import globals
from functions.globals import labelTypes
from functions.io_utils import load_locations, check_feat_dir, check_img_dir, check_label_dir
from functions.feat_utils import checkSignalLabels, convertBipolar2UnipolarBasic, extractFeatures
from functions.utils import toc, tic
from functions.gen_image import gen_images, gen_images_basic


def gen_features(windowLength, overlap):

    # check feature dataset directory
    feat_dir = check_feat_dir(windowLength, overlap)

    # EEG Files
    recordList = np.squeeze(scipy.io.loadmat(globals.file_records)['records'])
    for records, r in zip(recordList,range(len(recordList))):
        feat_file_name = feat_dir + '/' + 'record_' + str(r) + '.mat'
        if not os.path.isfile(feat_file_name):
            f = pyedflib.EdfReader(globals.dataset_link + np.str(records[0][0][0][0]))
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()

            # check eeg records labels
            # if more than 3 signal absent don't generate feats
            if checkSignalLabels(signal_labels) > 3:
                continue

            # read signal
            fs = np.int(f.getSampleFrequencies()[0])
            sigbufs = np.zeros((n, f.getNSamples()[0]))
            for i in np.arange(n):
                sigbufs[i, :] = f.readSignal(i, start=0, n=f.getNSamples()[0])
            f._close()
            del f

            # convert bipolar to unipolar
            unipolar_signal = convertBipolar2UnipolarBasic(signal_labels, sigbufs)

            # extractFeatures
            tic()
            features = extractFeatures(unipolar_signal, fs, windowLength, overlap)
            toc()
            scipy.io.savemat(feat_file_name, {'features': np.asarray(features)})


def gen_images_from_features(windowLength, overlap, pixelCount=[8, 8], genType='basic', method='RBF', edgeless=True):

    # Load Locations
    if genType == 'basic':
        locs_2d = load_locations(withProjection=False)
    elif genType == 'advanced':
        locs_2d = load_locations(withProjection=True)
    elif genType == 'none':
        return

    # check feature dataset directory
    feat_dir = check_feat_dir(windowLength, overlap)

    # check feature dataset directory
    img_dir = check_img_dir(windowLength, overlap, genType=genType, pixelCount=pixelCount[0], genMethod=method,
                            edgeless=edgeless)

    # EEG Files
    recordList = np.squeeze(scipy.io.loadmat(globals.file_records)['records'])



    # Load EEG Features
    for records, r in zip(recordList, range(len(recordList))):
        feat_file_name = feat_dir + '/' + 'record_' + str(r) + '.mat'
        if os.path.isfile(feat_file_name):
            img_file_name = img_dir + '/' + 'img_record_' + str(r) + '.mat'
            if not os.path.isfile(img_file_name):
                features = scipy.io.loadmat(feat_file_name)['features']
                # gen images
                tic()
                if genType == 'basic':
                    images_timewin = gen_images_basic(np.array(locs_2d), features, normalize=False)
                elif genType == 'advanced':
                    images_timewin = gen_images(np.array(locs_2d), features, pixelCount[0], normalize=True,
                                                edgeless=edgeless, method=method, multip=True)
                toc()
                scipy.io.savemat(img_file_name, {'images_timewin': np.asarray(images_timewin)})


def gen_labels(windowLength, overlap, preictalLen, postictalLen, minPreictalLen=0, minInterictalLen=0,
               excludedLen=0, sphLen=0):

    lbl_dir = check_label_dir(windowLength, overlap, preictalLen, postictalLen, minPreictalLen, minInterictalLen,
                              excludedLen, sphLen)

    # check feature dataset directory
    feat_dir = check_feat_dir(windowLength, overlap)

    preictalLen = preictalLen * 60
    postictalLen = postictalLen * 60
    minPreictalLen = minPreictalLen * 60
    minInterictalLen = minInterictalLen * 60
    excludedLen = excludedLen * 60
    sphLen = sphLen * 60

    # EEG Files
    subject = 0
    subject_list = []
    timeGapPre = 0
    recordList = np.squeeze(scipy.io.loadmat(globals.file_records)['records'])
    for records, r in zip(recordList, range(len(recordList))):
        feat_file_name = feat_dir + '/' + 'record_' + str(r) + '.mat'
        if os.path.isfile(feat_file_name):
            if subject != records[1][0][0]:
                subject = records[1][0][0]
                subject_list.append([])
                timeGapPre = 0
            fileName = records[0][0][0][0]
            isSeizured = records[2][0][0]
            length = records[5][0][0]
            timeGap = records[6][0][0] + timeGapPre
            timeGapPre = 0
            labels = np.full((length), labelTypes["interictal"])
            # step_length = np.int(window_length * (1 - overlap))
            # feature_len = np.int((length - step_length) / step_length)
            # labels = np.full((feature_len), labelTypes["interictal"])

            # fileName, isSeizured, length, timeGap, initial labels
            subject_list[records[1][0][0]-1].append([fileName, isSeizured, length, timeGap, labels, r])
        else:
            timeGapPre = records[6][0][0] + records[5][0][0] + timeGapPre

    seizureList = np.squeeze(scipy.io.loadmat(globals.file_seizures)['seizureList'])
    seizureListFiles = [seizure[0][0] for seizure in seizureList]

    # find ictal labels
    for subject in subject_list:
        for records, r in zip(subject, range(len(subject))):
            if records[1] and records[0] in seizureListFiles:
                for index in [index for index, fileName in enumerate(seizureListFiles) if fileName == records[0]]:
                    seizure_start = seizureList[index][2][0][0]-1
                    seizure_end = seizureList[index][3][0][0]-1
                    subject[r][4][seizure_start:seizure_end] = labelTypes["ictal"]

    # find other labels
    for subject, s in zip(subject_list, range(len(subject_list))):
        subject_file_name = lbl_dir + '/' + 'subject_' + str(s) + '.mat'
        if not os.path.isfile(subject_file_name):
            subject_ = globals.SubjectClass()

            # concatenate all labels
            subject_all_labels = []
            subject_record_idx = []
            label_idx = 0
            for records, r in zip(subject, range(len(subject))):
                timeGapLabels = np.full((records[3]), labelTypes["interictal"])
                subject_all_labels = np.hstack((subject_all_labels, timeGapLabels, records[4]))
                label_idx = label_idx + records[3]
                subject_record_idx.append([label_idx, label_idx+records[2]])
                label_idx = label_idx + records[2]

            # find postictal labels
            for l in range(1,len(subject_all_labels)):
                if subject_all_labels[l-1] == labelTypes["ictal"] and subject_all_labels[l] == labelTypes["interictal"]:
                    post_len = postictalLen
                    if l + post_len > len(subject_all_labels):
                        post_len = len(subject_all_labels) - l
                    if labelTypes["ictal"] in subject_all_labels[l:l + post_len]:
                        post_len = np.squeeze(np.where(subject_all_labels[l:l + post_len] == labelTypes["ictal"]))[0]
                    subject_all_labels[l:l + post_len] = labelTypes["postictal"]

            # find sph labels
            if sphLen:
                for l in range(len(subject_all_labels) - 1, 0, -1):
                    if subject_all_labels[l - 1] == labelTypes["interictal"] \
                            and subject_all_labels[l] == labelTypes["ictal"]:
                        sph_len = sphLen
                        if l - sph_len < 0:
                            sph_len = l
                        if labelTypes["postictal"] in subject_all_labels[l - sph_len:l]:
                            sph_len = sph_len - np.squeeze(
                                np.where(subject_all_labels[l - sph_len:l] == labelTypes["postictal"]))[-1] - 1
                        subject_all_labels[l - sph_len:l] = labelTypes["sph"]

            # find preictal labels
            for l in range(len(subject_all_labels)-1,0,-1):
                if subject_all_labels[l - 1] == labelTypes["interictal"] and\
                        (subject_all_labels[l] == labelTypes["ictal"] or subject_all_labels[l] == labelTypes["sph"]):
                    pre_len = preictalLen
                    if l - pre_len < 0:
                        pre_len = l
                    if labelTypes["postictal"] in subject_all_labels[l-pre_len:l]:
                        pre_len = pre_len - np.squeeze(np.where(subject_all_labels[l-pre_len:l] == labelTypes["postictal"]))[-1] - 1
                    subject_all_labels[l-pre_len:l] = labelTypes["preictal"]

            # find excluded labels
            if excludedLen:
                for l in range(len(subject_all_labels) - 1, 0, -1):
                    if subject_all_labels[l - 1] != labelTypes["ictal"] and subject_all_labels[l] == labelTypes["ictal"]:
                        exc_len = excludedLen
                        if l - exc_len < 0:
                            exc_len = l
                        idx_ = subject_all_labels[l - exc_len:l] == labelTypes["interictal"]
                        subject_all_labels[l - exc_len:l][idx_] = labelTypes["excluded"]
                for l in range(1, len(subject_all_labels)):
                    if subject_all_labels[l - 1] == labelTypes["ictal"] and subject_all_labels[l] != labelTypes["ictal"]:
                        exc_len = excludedLen
                        if l + exc_len > len(subject_all_labels):
                            exc_len = len(subject_all_labels) - l
                        idx_ = subject_all_labels[l:l + exc_len] == labelTypes["interictal"]
                        subject_all_labels[l:l + exc_len][idx_] = labelTypes["excluded"]

            # split all labels to records
            for r in range(len(subject)):
                subject[r][4] = subject_all_labels[subject_record_idx[r][0]:subject_record_idx[r][1]]


            # find feature labels
            step_length = np.int(windowLength * (1 - overlap))
            for records, r in zip(subject, range(len(subject))):
                feature_label = []
                feature_len = records[2] - (records[2] % step_length)
                for w in range(0, feature_len-step_length, step_length):
                    feature_label = np.hstack((feature_label, np.int(np.max(records[4][w:w+step_length]))))
                subject[r][4] = feature_label


            # concatenate all labels without gap
            subject_all_labels = []
            for records, r in zip(subject, range(len(subject))):
                subject_all_labels = np.hstack((subject_all_labels, records[4]))
            subject_all_labels= subject_all_labels.astype(int)

            # create index of label parts
            # interictal, preictal, ictal, postictal
            label_cur_idx= [1, 1, 1, 1, 1, 1]
            subject_all_label_idx = []
            for s in range(len(subject_all_labels)):
                label_idx = [0 ,0, 0, 0, 0, 0]
                label_idx[subject_all_labels[s]] = label_cur_idx[subject_all_labels[s]]
                if s < len(subject_all_labels) - 1:
                    if subject_all_labels[s] != subject_all_labels[s+1]:
                        label_cur_idx[subject_all_labels[s]] += 1
                subject_all_label_idx.append(label_idx)
            subject_all_label_idx = np.asarray(subject_all_label_idx)

            # interictal, preictal, ictal, postictal, excluded
            label_parts = []
            for i in range(subject_all_label_idx.shape[1]):
                part_ids = np.delete(np.unique(subject_all_label_idx[:, i]), 0)
                parts = []
                for part in part_ids:
                    idx = np.squeeze(np.where(subject_all_label_idx[:, i] == part))
                    parts.append(idx)
                label_parts.append(parts)

            interictal_parts = label_parts[0]
            preictal_parts = label_parts[1]
            excluded_parts = label_parts[4]

            # check preictal length
            if minPreictalLen:
                del_ids = []
                for p in range(len(preictal_parts)):
                    if len(preictal_parts[p])*step_length < minPreictalLen:
                        del_ids.append(p)
                preictal_parts = np.delete(preictal_parts, del_ids)

            # check interictal length
            if minInterictalLen:
                del_ids = []
                for p in range(len(interictal_parts)):
                    if len(interictal_parts[p])*step_length < minInterictalLen:
                        del_ids.append(p)
                interictal_parts = np.delete(interictal_parts, del_ids)

            subject_.records = subject
            subject_.all_labels = subject_all_labels
            subject_.interictal_parts = interictal_parts
            subject_.preictal_parts = preictal_parts
            subject_.excluded_parts = excluded_parts

            # save to file
            pickle_out = open(subject_file_name, "wb")
            pickle.dump(subject_, pickle_out)
            pickle_out.close()

        # pickle_in = open(subject_file_name, "rb")
        # subject_loaded = pickle.load(pickle_in)


def gen_data(featureParams, timingParams):

    # generate eeg features
    gen_features(featureParams.windowLength, featureParams.overlap)

    # generate images from features
    gen_images_from_features(featureParams.windowLength, featureParams.overlap, pixelCount=featureParams.pixelCount,
                             genType=featureParams.genType, method=featureParams.genMethod,
                             edgeless=featureParams.edgeless)

    # generate sample labels
    gen_labels(featureParams.windowLength, featureParams.overlap, timingParams.preictalLen, timingParams.postictalLen,
               minPreictalLen=timingParams.minPreictalLen,  minInterictalLen=timingParams.minInterictalLen,
               excludedLen=timingParams.excludedLen, sphLen=timingParams.sphLen)