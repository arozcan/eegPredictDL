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
    trainData.Xfold = fold[0]
    trainData.Xidx = np.unique(np.stack([trainData.Xfold + i for i in range(seqWinCount)]).ravel())
    trainData.Xlut = dict({i: j for i, j in zip(trainData.Xidx, range(len(trainData.Xidx)))})
    trainData.X = data[trainData.Xidx]
    trainData.Y = labels[trainData.Xfold]
    trainData.W = weights[trainData.Xfold ]
    trainData.SWC = seqWinCount

    validData = Data()
    validData.Xfold = fold[1]
    validData.Xidx = np.unique(np.stack([validData.Xfold + i for i in range(seqWinCount)]).ravel())
    validData.Xlut = dict({i: j for i, j in zip(validData.Xidx, range(len(validData.Xidx)))})
    validData.X = data[validData.Xidx]
    validData.Y = labels[validData.Xfold]
    validData.SWC = seqWinCount

    testData = Data()
    testData.Xfold = fold[2]
    testData.Xidx = np.unique(np.stack([testData.Xfold + i for i in range(seqWinCount)]).ravel())
    testData.Xlut = dict({i: j for i, j in zip(testData.Xidx, range(len(testData.Xidx)))})
    testData.X = data[testData.Xidx]
    testData.Y = labels[testData.Xfold]
    testData.SWC = seqWinCount

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

    inputLen = trainData.Xfold.shape[0]
    assert inputLen == len(trainData.Y)
    if shuffle:
        indices = np.arange(inputLen)
        np.random.shuffle(indices)
    for start_idx in range(0, inputLen, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if trainData.SWC > 1:
            Xdata = np.stack(np.stack(
                    trainData.X[trainData.Xlut[i+j]] for i in trainData.Xfold[excerpt]) for j in range(trainData.SWC))
            yield Xdata, trainData.Y[excerpt], trainData.W[excerpt]
        else:
            Xdata=np.stack(trainData.X[trainData.Xlut[i]] for i in trainData.Xfold[excerpt])
            yield Xdata, trainData.Y[excerpt], trainData.W[excerpt]


def iterate_minibatches_test(testData, batchsize, shuffle=False, index=None):
    """
    Iterates over the samples returing batches of size batchsize.
    """
    multiWin = True if testData.X.shape[0] < testData.X.shape[1] else False

    if index:
        input = testData.Xfold[index]
        target = testData.Y[index]
    else:
        input = testData.Xfold
        target = testData.Y
    inputLen = input.shape[0]
    assert inputLen == len(target)
    if shuffle:
        indices = np.arange(inputLen)
        np.random.shuffle(indices)
    for start_idx in range(0, inputLen, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if testData.SWC > 1:
            Xdata = np.stack(
                np.stack(testData.X[testData.Xlut[i + j]] for i in input[excerpt]) for j in range(testData.SWC))
            yield Xdata, target[excerpt]
        else:
            Xdata = np.stack(testData.X[testData.Xlut[i]] for i in input[excerpt])
            yield Xdata, target[excerpt]



def train_with_iterate_minibatches(trainFn, trainData, trainParams):
    trainErr = 0
    trainAcc = 0
    for batch in iterate_minibatches_train(trainData, trainParams.batchSize, shuffle=False):
        inputs, targets, weights = batch
        err, acc = trainFn(inputs, targets, weights, trainParams.l2)
        trainErr += err * len(targets)
        trainAcc += acc * len(targets)
    trainErr = trainErr / len(trainData.Y)
    trainAcc = trainAcc / len(trainData.Y)
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
        testLen = 0
        for batch in iterate_minibatches_test(testData, trainParams.batchSize, shuffle=False, index=index):
            inputs, targets = batch
            err, acc, pred = testFn(inputs, targets)
            testErr += err * len(targets)
            testAcc += acc * len(targets)
            testLen += len(targets)
            testPred = pred if testPred == [] else np.vstack((testPred, pred))
        testErrAll += testErr / testLen
        testAccAll += testAcc / testLen
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
        if testData.SWC > 1:
            input = np.stack(
                np.stack(testData.X[testData.Xlut[i + j]] for i in testData.Xfold[index]) for j in range(testData.SWC))
        else:
            input = np.stack(testData.X[testData.Xlut[i]] for i in testData.Xfold[index])
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

