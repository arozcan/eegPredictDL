# -*- coding: utf-8 -*-
from __future__ import print_function

from functions import globals
from functions.gen_data import gen_data
from functions.params import init_model_parameters, init_feature_types

if __name__ == '__main__':

    """
    Define Parameters
    """
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

    modelParams =init_model_parameters(modelName[10], pixelCount[2])

    # feature parameters
    featureParams = globals.FeatureParams()
    featureParams.pixelCount = modelParams.pixelCount
    featureParams.genType = modelParams.genImageType
    featureParams.featureTypes = init_feature_types(psd=True, moment=True, hjorth=True)

    # timing parameters
    timingParams = globals.TimingParams()

    # train data parameters
    trainDataParams = globals.TrainDataParams()
    trainDataParams.seqWinCount = modelParams.seqWinCount
    trainDataParams.subject = 0  # patient id

    # train parameters
    trainParams = globals.TrainParams()
    trainParams.onlineWeights = False
    trainParams.l2 = 0.01
    trainParams.plotPrediction = True
    trainParams.learnRate = 0.001

    """
    Generate Data and Prediction
    """

    # prepare dataset
    #gen_data(featureParams, timingParams)

    # predict data
    from functions.predict_data import predict_data
    predict_data(featureParams, timingParams, trainDataParams, trainParams, modelParams)
