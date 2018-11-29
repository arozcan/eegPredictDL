# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import pickle
import sqlite3
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

def check_img_dir(windowLength, overlap, genImageType, pixelCount=8, genMethod='RBF', edgeless=True):
    # is dir exist
    dir = globals.file_img_dataset + str(windowLength) + 's_' + str(overlap) + 'o_' + str(genImageType)
    if genImageType == 'advanced':
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


def check_test_db(resultFeats):

    db_name = globals.file_test_db

    if not os.path.isfile(db_name):

        db = sqlite3.connect(db_name)
        cursor = db.cursor()

        sql = """CREATE TABLE IF NOT EXISTS test(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, """
        sql += """sqltime TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL"""
        keys = []
        for groups in globals.dbFeatGroups:
            for key in resultFeats.get(groups).__dict__.keys():
                if (key not in keys) and (key in globals.dbFeats):
                    keys.append(key)
                    sql += ", " + str(key)
                    val = resultFeats.get(groups).__dict__.get(key)
                    if isinstance(val, basestring) or isinstance(val, np.ndarray) or isinstance(val, list) :
                        sql += " text"
                    elif isinstance(val, bool) or isinstance(val, int) or isinstance(val, dict):
                        sql += " integer"
                    elif isinstance(val, float):
                        sql += " real"
        sql += " )"
        cursor.execute(sql)
        db.commit()
        db.close()


def insert_to_test_db(resultFeats):

    db_name = globals.file_test_db

    if os.path.isfile(db_name):
        db = sqlite3.connect(db_name)
        cursor = db.cursor()

        sql = "INSERT INTO test ("
        keys = []
        vals = []
        for groups in globals.dbFeatGroups:
            for key in resultFeats.get(groups).__dict__.keys():
                if (key not in keys) and (key in globals.dbFeats):
                    if len(keys):
                        sql += ", "
                    keys.append(key)
                    sql += str(key)
                    val = resultFeats.get(groups).__dict__.get(key)
                    if isinstance(val, basestring):
                        val = "'" +str(val)+ "'"
                    elif isinstance(val, np.ndarray) or isinstance(val, list):
                        val = "'" + str(val).replace(" ", "") + "'"
                    elif isinstance(val, bool) or isinstance(val, int):
                        val = int(val)
                    elif isinstance(val, dict):
                        val = int(val["active"])
                    elif isinstance(val, float):
                        val = float(val)
                    vals.append(val)
        sql += ") VALUES ("
        for val, val_idx in zip(vals, range(len(vals))):
            if val_idx > 0:
                sql += ", "
            sql += str(val)
        sql += ")"
        testId = cursor.execute(sql).lastrowid
        db.commit()
        db.close()
        return testId
    else:
        exit('Test database not found!')


def search_in_test_db(resultFeats):

    db_name = globals.file_test_db

    if os.path.isfile(db_name):
        db = sqlite3.connect(db_name)
        cursor = db.cursor()

        sql = "SELECT id FROM test WHERE "
        keys = []
        vals = []
        for groups in globals.dbFeatGroups:
            for key in resultFeats.get(groups).__dict__.keys():
                if (key not in keys) and (key in globals.dbFeats):
                    if len(keys):
                        sql += " AND "
                    keys.append(key)
                    sql += str(key) + "="
                    val = resultFeats.get(groups).__dict__.get(key)
                    if isinstance(val, basestring):
                        sql += "'" + str(val) + "'"
                    elif isinstance(val, np.ndarray) or isinstance(val, list):
                        sql += "'" + str(val).replace(" ", "") + "'"
                    elif isinstance(val, bool) or isinstance(val, int):
                        sql += str(int(val))
                    elif isinstance(val, dict):
                        sql += str(int(val["active"]))
                    elif isinstance(val, float):
                        sql += str(float(val))
                    vals.append(val)
        cursor.execute(sql)
        rows = cursor.fetchall()
        db.commit()
        db.close()
        return rows
    else:
        print('Test database not found!')
        return []


def get_from_test_db(feats):

    db_name = globals.file_test_db

    if os.path.isfile(db_name):
        db = sqlite3.connect(db_name)
        cursor = db.cursor()

        sql = "SELECT id FROM test"
        if feats:
            sql += " WHERE "
            for feat, idx in zip(feats.keys(), range(len(feats.keys()))):
                if idx:
                    sql += " AND "
                sql += str(feat) + "="
                val = feats.get(feat)
                if isinstance(val, basestring):
                    sql += "'" +str(val)+ "'"
                elif isinstance(val, np.ndarray) or isinstance(val, list):
                    sql += "'" + str(val).replace(" ", "") + "'"
                elif isinstance(val, bool) or isinstance(val, int):
                    sql += str(int(val))
                elif isinstance(val, dict):
                    sql += str(int(val["active"]))
                elif isinstance(val, float):
                    sql += str(float(val))
        cursor.execute(sql)
        rows = cursor.fetchall()
        db.commit()
        db.close()
        return rows
    else:
        exit('Test database not found!')


def save_results(trainResult, featureParams, timingParams, trainDataParams, trainParams, modelParams, dataset):

    trainParams.callbacks=None
    modelParams.networkSharedVars = None
    dataset.data = None
    results = {"trainResult": trainResult,
               "featureParams": featureParams,
               "timingParams": timingParams,
               "trainDataParams": trainDataParams,
               "trainParams": trainParams,
               "modelParams": modelParams,
               "dataset": dataset}

    check_test_db(results)
    test_id = insert_to_test_db(results)

    # save results and params
    test_name = globals.file_test + "test_" + str(test_id) + ".mat"
    pickle_out = open(test_name, "wb")
    pickle.dump(results, pickle_out)
    pickle_out.close()


def load_results(test_id):

    test_name = globals.file_test + "test_" + str(test_id) + ".mat"
    pickle_in = open(test_name, "rb")
    results = pickle.load(pickle_in)
    pickle_in.close()
    return results


def check_test_results(featureParams, timingParams, trainDataParams, trainParams, modelParams):

    params = {"featureParams": featureParams,
              "timingParams": timingParams,
              "trainDataParams": trainDataParams,
              "trainParams": trainParams,
              "modelParams": modelParams}

    return search_in_test_db(params)
