# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from functions.globals import Data


def swap_data_axes(data, ind1, ind2):
    swappedData = []
    for i in data:
        swappedData.append(np.swapaxes(np.asarray(i), ind1, ind2))
    return swappedData


def move_data_axes(data, ind1, ind2):
    movedData = []
    for i in data:
        movedData.append(np.moveaxis(np.asarray(i), ind1, ind2))
    return movedData


def transform_data(data, sc):
    transformedData = []
    for d in data:
        d_temp = d.reshape(-1)
        d_temp[~np.isnan(d_temp)] = sc.transform(d_temp[~np.isnan(d_temp)].reshape(-1, 1)).reshape(-1)
        d_temp = d_temp.reshape(d.shape)
        transformedData.append(d_temp)
    return transformedData


def scale_data(trainData, validData, testData, withMean=False, fitAllTrainData=False):

    sc = StandardScaler(with_mean=withMean)

    # oznitelik boyunca standartizasyon yapabilek icin boyut sirasini degistiriyoruz
    trainData, validData, testData = swap_data_axes((trainData, validData, testData), 0, 2)

    if fitAllTrainData:
        trainAll = np.concatenate((trainData, validData), axis=1)
    else:
        trainAll = trainData

    for i in range(len(trainAll)):
        # train verilerine gore standartizasyon parametreleri hesaplandi
        train_temp = trainAll[i].reshape(-1)
        sc.fit(train_temp[~np.isnan(train_temp)].reshape(-1, 1))

        # veriler standartize ediliyor
        trainData[i], validData[i], testData[i] = transform_data((trainData[i], validData[i], testData[i]), sc)

    # boyut sirasini eski haline getiriyoruz
    trainData, validData, testData = swap_data_axes((trainData, validData, testData), 0, 2)

    return trainData, validData, testData


def reformat_data(fold, data, labels, weights, seqWinCount, scaleData=True, withMean=False, fitAllTrainData = False):
    trainData = Data()
    trainData.X = np.asarray([data[fold[0] + i] for i in range(seqWinCount)])
    trainData.Y = labels[fold[0]]
    trainData.W = weights[fold[0]]

    validData = Data()
    validData.X = np.asarray([data[fold[1] + i] for i in range(seqWinCount)])
    validData.Y = labels[fold[1]]

    testData = Data()
    testData.X = np.asarray([data[fold[2] + i] for i in range(seqWinCount)])
    testData.Y = labels[fold[2]]

    # Scale data by using mean and variance of features
    if scaleData:
        trainData.X, validData.X, testData.X = scale_data(trainData.X, validData.X, testData.X,
                                                          fitAllTrainData=fitAllTrainData, withMean=withMean)

    trainData.X = np.squeeze(np.nan_to_num(trainData.X)).astype("float32", casting='unsafe')
    validData.X = np.squeeze(np.nan_to_num(validData.X)).astype("float32", casting='unsafe')
    testData.X = np.squeeze(np.nan_to_num(testData.X)).astype("float32", casting='unsafe')

    return trainData, validData, testData


def iterate_minibatches_train(trainData, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    """
    multiWin = True if trainData.X.shape[0] < trainData.X.shape[1] else False

    if not multiWin:
        inputLen = trainData.X.shape[0]
    else:
        inputLen = trainData.X.shape[1]
    assert inputLen == len(trainData.Y)
    if shuffle:
        indices = np.arange(inputLen)
        np.random.shuffle(indices)
    for start_idx in range(0, inputLen, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if not multiWin:
            yield trainData.X[excerpt], trainData.Y[excerpt], trainData.W[excerpt]
        else:
            yield trainData.X[:, excerpt], trainData.Y[excerpt], trainData.W[excerpt]


def iterate_minibatches_test(testData, batchsize, shuffle=False, index=None):
    """
    Iterates over the samples returing batches of size batchsize.
    """
    multiWin = True if testData.X.shape[0] < testData.X.shape[1] else False

    if not multiWin:
        if index:
            input = testData.X[index]
            target = testData.Y[index]
        inputLen = input.shape[0]
    else:
        if index:
            input = testData.X[:, index]
            target = testData.Y[index]
        inputLen = input.shape[1]
    assert inputLen == len(target)
    if shuffle:
        indices = np.arange(inputLen)
        np.random.shuffle(indices)
    for start_idx in range(0, inputLen, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if not multiWin:
            yield input[excerpt], target[excerpt]
        else:
            yield input[:, excerpt], target[excerpt]


def train_with_iterate_minibatches(trainFn, trainData, trainParams):
    trainErr = 0
    trainAcc = 0
    trainBatches = 0
    for batch in iterate_minibatches_train(trainData, trainParams.batchSize, shuffle=False):
        inputs, targets, weights = batch
        err, acc = trainFn(inputs, targets, weights, trainParams.l2)
        trainErr += err
        trainAcc += acc
        trainBatches += 1
    trainErr = trainErr / trainBatches
    trainAcc = trainAcc / trainBatches
    return trainErr, trainAcc


def test_with_iterate_minibatches(testFn, testData, trainParams):
    testErrAll = 0
    testAccAll = 0
    testPredAll = []
    u, u_index = np.unique(testData.Y, return_index=True)
    u_index = np.append(u_index, len(testData.Y))
    u_index = [range(u_index[i], u_index[i + 1]) for i in u]
    for index in u_index:
        testErr = 0
        testAcc = 0
        testPred = []
        testBatches = 0
        for batch in iterate_minibatches_test(testData, trainParams.batchSize, shuffle=False, index=index):
            inputs, targets = batch
            err, acc, pred = testFn(inputs, targets)
            testErr += err
            testAcc += acc
            testBatches += 1
            testPred = pred if testPred == [] else np.vstack((testPred, pred))
        testErrAll += testErr / testBatches
        testAccAll += testAcc / testBatches
        testPredAll.append(testPred)
    testErrAll = testErrAll / len(u_index)
    testAccAll = testAccAll / len(u_index)
    return testErrAll, testAccAll, testPredAll


def test_per_label(testFn, testData):
    testErr = 0
    testAcc = 0
    testPred = []
    u, u_index = np.unique(testData.Y, return_index=True)
    u_index = np.append(u_index, len(testData.Y))
    u_index = [range(u_index[i], u_index[i + 1]) for i in u]
    multiWin = True if testData.X.shape[0] < testData.X.shape[1] else False
    for index in u_index:
        if not multiWin:
            input = testData.X[index]
        else:
            input = testData.X[:, index]
        err, acc, pred = testFn(input, testData.Y[index])
        testErr += err
        testAcc += acc
        testPred.append(pred)
    testErr = testErr / len(u_index)
    testAcc = testAcc / len(u_index)
    return testErr, testAcc, testPred


def print_epoch(epochParams, state='begin'):
    dict = {0: ['epoch', 'epoch       ', '{:3}         '],
            1: ['trainErr', 'train loss  ', '{:.6f}     '],
            2: ['validErr', 'valid loss  ', '{:.6f}     '],
            3: ['trainValRatio', 'train/valid ', '{:.3f}        '],
            4: ['trainAcc', 'train accu  ', '{:3.3f}%       '],
            5: ['validAcc', 'valid accu  ', '{:3.3f}%       '],
            6: ['duration', 'duration    ', '{:.3f}s       ']}
    text = ""
    if state == 'begin':
        line1 = "\n"
        line2 = "\n"
        for i in dict:
            line1 += dict[i][1] + '\t'
            line2 += '------------' + '\t'
        text += line1 + line2
    elif state == 'step':
        line1 = ""
        for i in dict:
            line1 += dict[i][2].format(epochParams.__getattribute__(dict[i][0])[-1]) + '\t'
        text += line1
    elif state == 'end':
        print("")
    print(text)


def print_epoch(epochParams, state='begin'):
    dict = {0: ['epoch', 'epoch       ', '{:3}         '],
            1: ['trainErr', 'train loss  ', '{:.6f}     '],
            2: ['validErr', 'valid loss  ', '{:.6f}     '],
            3: ['trainValRatio', 'train/valid ', '{:.3f}        '],
            4: ['trainAcc', 'train accu  ', '{:3.3f}%       '],
            5: ['validAcc', 'valid accu  ', '{:3.3f}%       '],
            6: ['duration', 'duration    ', '{:.3f}s       ']}
    text = ""
    if state == 'begin':
        line1 = "\n"
        line2 = "\n"
        for i in dict:
            line1 += dict[i][1] + '\t'
            line2 += '------------' + '\t'
        text += line1 + line2
    elif state == 'step':
        line1 = ""
        for i in dict:
            line1 += dict[i][2].format(epochParams.__getattribute__(dict[i][0])[-1]) + '\t'
        text += line1
    elif state == 'end':
        print("")
    print(text)

