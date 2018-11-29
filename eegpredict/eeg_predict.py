# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from functions import globals
from functions.gen_data import gen_data
from functions.params import init_model_parameters, init_feature_types, init_params
from functions.io_utils import get_from_test_db, load_results, check_test_results
from functions.plot_data import compare_subject_roc, compare_subject_prediction, plot_subject_roc, \
    plot_subject_prediction, compare_various_result
from functions.utils import print_params, merge_dicts
from functions.predict_data import predict_data
import numpy as np
import itertools

# model parameters
modelName = {0: 'model_svm',
             1: 'model_custom_mlp',
             2: 'model_custom_mlp_multi',
             3: 'model_cnn_basic',
             4: 'model_cnn',
             5: 'model_cnn_max',
             6: 'model_cnn_conv1d',
             7: 'model_cnn_lstm',
             8: 'model_cnn_mix',
             9: 'model_cnn_lstm_hybrid',
             10: 'model_cnn3d',
             11: 'model_cnn3d_new'}

pixelCount = {0: 20,
              1: [4, 5],
              2: [8, 8],
              3: [16, 16]}


def do_job(job, **jobParams):

    if job == "predict":

        """
        Define Parameters
        """
        predictParams = jobParams.get("predictParams")

        # init model parameters
        modelParams, jobParams = init_model_parameters(predictParams)

        # init feature parameters
        featureParams = init_params(globals.FeatureParams(), predictParams)

        # init timing parameters
        timingParams = init_params(globals.TimingParams(), predictParams)

        # init train data parameters
        trainDataParams = init_params(globals.TrainDataParams(), predictParams)

        # init train parameters
        trainParams = init_params(globals.TrainParams(), predictParams)

        """
        Generate Data and Prediction
        """

        # prepare dataset
        gen_data(featureParams, timingParams)

        # predict data
        if len(check_test_results(featureParams, timingParams, trainDataParams, trainParams, modelParams)):
            print("Trained with these parameters before!")
            print_params(featureParams, timingParams, trainDataParams, trainParams, modelParams)
        else:
            predict_data(featureParams, timingParams, trainDataParams, trainParams, modelParams)

    if job == "compare":

        compareSet = jobParams.get("compareSet")
        print("Subject: {}".format(compareSet[0].get("subject")))
        plotTitle = jobParams.get("plotTitle")
        plotLabels = jobParams.get("plotLabels")

        results = []
        for feats in compareSet:
            ids = get_from_test_db(feats)
            for id in ids:
                results.append(load_results(str(id[0])))
        if results:
            compare_subject_roc(results, plotLabels, plotTitle)
            compare_subject_prediction(results, plotLabels, plotTitle)

    if job == "compare_various":
        compareParams = jobParams.get("compareParams")
        validParams = jobParams.get("validParams")
        selectSet = jobParams.get("selectSet")
        compare_various_result(compareParams, validParams, selectSet)

    if job == "plot":
        plotSet = jobParams.get("plotSet")
        plotTitle = jobParams.get("plotTitle")
        plotLabels = jobParams.get("plotLabels")

        results = []
        for feats in plotSet:
            ids = get_from_test_db(feats)
            for id in ids:
                results.append(load_results(id[0]))

        for result in results:
            plot_subject_roc(result, plotTitle)
            plot_subject_prediction(result, plotTitle)


if __name__ == '__main__':

    # subject list
    subjectList = np.array([1, 2, 3, 5, 7, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21, 23])


    # Train with single parameters
    trainWithSingleParams = False
    if trainWithSingleParams:
        do_job(job="predict", predictParams={"subject": 18, "modelName": modelName[4], "pixelCount": pixelCount[2],
                                             "preictalLen": 60, "excludedLen": 240, "onlineWeights": True, "l2": 0.01,
                                             "earlyStopping": True, "adaptiveL2": False, "evalExcluded": False,
                                             "minPreictalLen":5, "postictalLen":10})


    #Plot train results
    plotResults = False
    if plotResults:
        do_job(job="plot", plotTitle= ["subject", "modelName", "preictalLen"],
               plotSet=[{"modelName": modelName[10], "pixelCount": pixelCount[2], "subject": 18, "excludedLen": 240, "evalExcluded": False}])


    # Train with various parameters
    trainWithVariousParams = False
    if trainWithVariousParams:
        subjectList = [{"subject": subject} for subject in subjectList]

        modelList = [{"modelName": modelName[1], "pixelCount": pixelCount[0]},
                     {"modelName": modelName[4], "pixelCount": pixelCount[2]},
                     {"modelName": modelName[10], "pixelCount": pixelCount[2]}]

        timingList = [{"preictalLen": 30, "excludedLen": 60, "onlineWeights": False},
                      {"preictalLen": 30, "excludedLen": 120, "onlineWeights": False},
                      {"preictalLen": 30, "excludedLen": 240, "onlineWeights": False},
                      {"preictalLen": 60, "excludedLen": 120, "onlineWeights": False},
                      {"preictalLen": 60, "excludedLen": 120, "onlineWeights": True},
                      {"preictalLen": 60, "excludedLen": 240, "onlineWeights": False},
                      {"preictalLen": 60, "excludedLen": 240, "onlineWeights": True}]

        trainList = [{"l2": 0, "earlyStopping": True, "adaptiveL2": False},
                     {"l2": 0.01, "earlyStopping": True, "adaptiveL2": False}]



        trialParams = itertools.product(*[timingList, trainList, subjectList, modelList])
        for trialParam in trialParams:
            do_job(job="predict", predictParams=merge_dicts(trialParam))

    # Compare train results
    compareResults = False
    if compareResults:
        for i in subjectList:
            do_job(job="compare", plotLabels=["modelName", "onlineWeights", "l2"], plotTitle=["subject"],
                   compareSet=[
                       {"modelName": modelName[1], "pixelCount": pixelCount[0], "subject": i, "excludedLen": 240,
                        "adaptiveL2": False, "onlineWeights": False, "preictalLen": 60},
                       {"modelName": modelName[4], "pixelCount": pixelCount[2], "subject": i, "excludedLen": 240,
                        "adaptiveL2": False, "onlineWeights": False, "preictalLen": 60},
                       {"modelName": modelName[10], "pixelCount": pixelCount[2], "subject": i, "excludedLen": 240,
                        "adaptiveL2": False, "onlineWeights": False, "preictalLen": 60}
                       ])

    # Compare with various parameters
    compareWithVariousParams = True
    if compareWithVariousParams:
        subjectList = [{"subject": subject} for subject in subjectList]

        modelList = [{"modelName": modelName[1], "pixelCount": pixelCount[0]},
                     {"modelName": modelName[4], "pixelCount": pixelCount[2]},
                     {"modelName": modelName[10], "pixelCount": pixelCount[2]}]

        timingList = [{"preictalLen": 30, "excludedLen": 60, "onlineWeights": False},
                      {"preictalLen": 30, "excludedLen": 120, "onlineWeights": False},
                      {"preictalLen": 30, "excludedLen": 240, "onlineWeights": False},
                      {"preictalLen": 60, "excludedLen": 120, "onlineWeights": False},
                      {"preictalLen": 60, "excludedLen": 120, "onlineWeights": True},
                      {"preictalLen": 60, "excludedLen": 240, "onlineWeights": False},
                      {"preictalLen": 60, "excludedLen": 240, "onlineWeights": True}]

        trainList = [{"l2": 0, "earlyStopping": True, "adaptiveL2": False},
                     {"l2": 0.01, "earlyStopping": True, "adaptiveL2": False}]



        compareParams = [timingList, trainList, subjectList, modelList]
        do_job(job="compare_various", compareParams=compareParams,
               validParams=["subject", "modelName", "preictalLen,excludedLen,onlineWeights", "l2"],
               selectSet=[ timingList,
                           subjectList
                           ])


    raw_input("Press Enter to continue...")



