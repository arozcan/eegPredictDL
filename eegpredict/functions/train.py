# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
from sklearn.metrics import accuracy_score, log_loss
from functions.utils import dashed_line
from functions.train_utils import reformat_data, train_with_iterate_minibatches, test_with_iterate_minibatches,\
    test_per_label, print_epoch
from functions.globals import TrainResult, EpochParams, TestParams
from functions.plot_data import plot_only_prediction
import functions.callback as clbk


def create_callbacks(model):

    _callbacks = [clbk.Standard()]

    if model.trainParams.earlyStopping['active']:
        earlyStopping = clbk.EarlyStopping(monitor=model.trainParams.earlyStopping['monitor'],
                                           patience=model.trainParams.earlyStopping['patience'],
                                           min_delta=model.trainParams.earlyStopping['min_delta'],
                                           )
        _callbacks.append(earlyStopping)

    if model.trainParams.adaptiveL2['active']:
        adaptiveL2 = clbk.AdaptiveL2(monitor=model.trainParams.adaptiveL2['monitor'],
                                     treshold=model.trainParams.adaptiveL2['treshold'],
                                     epoch=model.trainParams.adaptiveL2['epoch'],
                                     l2_current=model.trainParams.l2,
                                     l2_step=model.trainParams.adaptiveL2['l2_step']
                                     )
        _callbacks.append(adaptiveL2)

    model.trainParams.callbacks = clbk.CallbackList(_callbacks)

    model.trainParams.callbacks.set_model(model)




def check_pretrained_init(model):
    if model.train_params.pre_trained_init:
        net_param_val = \
        np.load(globals.file_model + 'weights_{}_pre_train_params.npz'.format(model.model_params.model_name))[
            'arr_0.npy']
        model.network_param_values = net_param_val


def fit_neural_network(model, trainData, validData):

    epochParams = EpochParams()

    print_epoch(epochParams, state='begin')

    # num_epoch sayisinca iterasyon
    while epochParams.curEpoch < model.trainParams.numEpoch:
        start_time = time.time()

        # on epoch begin callbacks
        model.trainParams.callbacks.on_epoch_begin(epochParams.curEpoch, epochParams, model.trainParams)

        # train with train data
        trainErr, trainAcc = train_with_iterate_minibatches(model.trainFn, trainData, model.trainParams)

        # test with validation data
        validErr, validAcc, validPred = test_per_label(model.valFn, validData)


        # save epoch params
        epochParams.epoch.append(epochParams.curEpoch+1)
        epochParams.duration.append(time.time() - start_time)
        epochParams.trainErr.append(trainErr)
        epochParams.trainAcc.append(trainAcc*100)
        epochParams.validErr.append(validErr)
        epochParams.validAcc.append(validAcc*100)
        epochParams.validPred.append(validPred)
        epochParams.trainValRatio.append(trainErr/validErr)
        epochParams.netParamVal.append(model.model.get_weights(model))

        print_epoch(epochParams, state='step')

        epochParams.curEpoch += 1

        # on epoch begin callbacks
        model.trainParams.callbacks.on_epoch_end(epochParams.curEpoch, epochParams, model.trainParams)

    print_epoch(epochParams, state='end')

    # find best validation error index
    # if model.trainParams.earlyStopMonitor == "validErr":
    #     bestIdx = np.argmin(epochParams.validErr)
    # elif model.trainParams.earlyStopMonitor == "validAcc":
    #     bestIdx = np.argmax(epochParams.validAcc)
    # print("Best train index:{}".format(bestIdx+1))
    #
    # # load network best weights
    # lasagne.layers.set_all_param_values(model.model, epochParams.netParamVal[bestIdx])
    epochParams.netParamVal = epochParams.netParamVal[epochParams.bestEpoch]
    epochParams.validPred = epochParams.validPred[epochParams.bestEpoch]

    return model, epochParams


def fit_svm(model, trainData, validData):

    # validate verileri train verilerine ekleniyor
    x_train = np.concatenate((trainData.X, validData.X))
    y_train = np.concatenate((trainData.Y, validData.Y))

    # randomize ediliyor
    trainIndices = range(len(x_train))
    np.random.shuffle(trainIndices)
    x_train = x_train[trainIndices]
    y_train = y_train[trainIndices]

    # oznitelikler tek boyutlu hale getiriliyor
    x_train = np.reshape(x_train, (x_train.shape[0], -1))

    # svm egitiliyor
    model.model.fit(x_train, y_train)

    return model


def fit_model(model, trainData, validData, trainResult):

    # on train begin callbacks
    model.trainParams.callbacks.on_train_begin(trainParams=model.trainParams)

    if model.modelParams.modelType == "neural":

        # fit network
        model, epochParams = fit_neural_network(model, trainData, validData)

        if model.trainParams.saveModelParams:
            trainResult.netWeights.append(epochParams.netParamVal)
        epochParams.netParamVal = None

    elif model.modelParams.modelType == "svm":
        # fit svm
        model = fit_svm(model, trainData, validData)
        epochParams = []

    # on train end callbacks
    model.trainParams.callbacks.on_train_end()

    trainResult.trainParams.append(epochParams)

    return model


def test_neural_network(model, testData):
    testParams = TestParams()
    # test with test data
    testParams.testErr, testParams.testAcc, testParams.testPred = test_with_iterate_minibatches(model.testFn, testData,
                                                                                                model.trainParams)
    return testParams


def test_svm(model, testData):
    testParams = TestParams()
    testErr = 0
    testAcc = 0
    testPred = []
    u, u_index = np.unique(testData.Y, return_index=True)
    u_index = np.append(u_index, len(testData.Y))
    u_index = [range(u_index[i], u_index[i + 1]) for i in u]

    # oznitelikler tek boyutlu hale getiriliyor
    testData.X = np.reshape(testData.X, (testData.X.shape[0], -1))
    for index in u_index:
        pred = model.model.predict(testData.X[index])
        acc = accuracy_score(testData.Y[index], pred)
        if testData.Y[index[0]] in pred:
            err = log_loss(pred, testData.Y[index])
        else:
            err = testData.Y[index[0]]
        testErr += err
        testAcc += acc
        testPred.append(pred)

    testParams.testErr = testErr / len(u_index)
    testParams.testAcc = testAcc / len(u_index)
    testParams.testPred = testPred

    return testParams


def test_model(model, testData, trainResult):

    if model.modelParams.modelType == "neural":
        testParams = test_neural_network(model, testData)
    elif model.modelParams.modelType == "svm":
        testParams = test_svm(model, testData)

    print("test loss: {:.6f}\t\ttest accuracy: {:3.3f} %".format(testParams.testErr, testParams.testAcc * 100) +
          dashed_line())

    # prediction sonuclarini cizdir
    if model.trainParams.plotPrediction:
        plot_only_prediction(testParams.testPred)

    trainResult.testParams.append(testParams)

    return

def cross_validation(model, dataset):

    trainResult = TrainResult()

    for sample, s in zip(dataset.fold_pairs, range(len(dataset.fold_pairs))):

        sampleResult = TrainResult()

        for fold, f in zip(sample, range(len(sample))):

            print('Sample: {}\tFold: {}\t\t'.format(s+1, f+1), end='')



            # get train, valid and test data
            trainData, validData, testData = reformat_data(fold, dataset.data, dataset.labels, dataset.weights,
                                                           model.modelParams.seqWinCount, scaleData=True,
                                                           withMean=True, fitAllTrainData=False)

            # train model
            model = fit_model(model, trainData, validData, sampleResult)

            # test model
            test_model(model, testData, sampleResult)

        trainResult.trainParams.append(sampleResult.trainParams)
        trainResult.testParams.append(sampleResult.testParams)
        trainResult.netWeights.append(sampleResult.netWeights)
    return trainResult


def train_model(model, dataset):

    print('Model training...' + dashed_line())

    create_callbacks(model)

    trainResult = cross_validation(model, dataset)

    return trainResult