import numpy as np
import matplotlib.pyplot as plt
from functions.math_utils import moving_average
from sklearn.metrics import roc_curve, auc
from functions.ranking import my_roc_curve
from functions.globals import labelTypes


def refractory_period_filter(predictions, labels, featureParams, period=30):

    stepLen = (featureParams.windowLength * (1 - featureParams.overlap))
    period = np.int(period * 60 / stepLen)

    predictions_new = np.zeros(len(predictions))
    for idx in range(len(predictions)):
        idx_end = idx + period
        if idx_end > len(predictions):
            idx_end = len(predictions)
        if np.max(predictions_new[idx:idx_end]) < predictions[idx]:
            labelTruth = labels[idx:idx_end] == labels[idx]
            predictions_new[idx:idx_end][labelTruth] = predictions[idx]

    return predictions_new


def plot_only_prediction(predict):
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Prediction Results', fontweight='bold')

    predict_int_avg=moving_average(predict[0], 30)
    axes[0].plot(predict_int_avg)
    axes[0].set_title('Interictal Predictions')
    axes[0].set_xlim(0, len(predict_int_avg))
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=max(predict_int_avg), color='red', linewidth=0.5, label=str(max(predict_int_avg)))
    axes[0].legend()

    predict_pre_avg = moving_average(predict[1], 30)
    axes[1].plot(predict_pre_avg)
    axes[1].set_title('Precital Predictions')
    axes[1].set_xlim(0, len(predict_pre_avg))
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=max(predict_pre_avg), color='red', linewidth=0.5, label=str(max(predict_pre_avg)))
    axes[1].legend()

    fig.show()


def plot_subject_prediction(testParams, foldPairs, labels):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Prediction Results', fontweight='bold')

    predictions = np.zeros((len(labels)), dtype=float)
    labels[labels != 1] = 0

    for s in range(len(testParams)):
        for f in range(len(testParams[s])):
            predictions[foldPairs[s][f][2]] += np.squeeze(np.hstack((np.squeeze(testParams[s][f].testPred[0]),
                                                                     np.squeeze(testParams[s][f].testPred[1]))))


    predictions = predictions / len(testParams[0])
    predictions=moving_average(predictions, 30)
    axes.plot(predictions)
    axes.plot(labels)
    axes.set_xlim(0, len(predictions))
    axes.set_ylim(0, 1.05)
    axes.legend()

    fig.show()


def plot_subject_scores(testParams, foldPairs, labels, featureParams):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Prediction Results', fontweight='bold')

    testPredictsAll = []
    testLabels = np.full(len(labels), fill_value=-1, dtype=int)

    for s in range(len(testParams)):
        testLabels[foldPairs[s][0][2]] = np.squeeze(
            np.hstack((np.full(len(testParams[s][0].testPred[0]),fill_value=labelTypes["interictal"]),
                       np.full(len(testParams[s][0].testPred[1]),fill_value=labelTypes["preictal"]))))

    for f in range(len(testParams[0])):
        testPredicts = np.zeros(len(labels), dtype=float)
        for s in range(len(testParams)):
            testPredicts[foldPairs[s][f][2]] += np.squeeze(
                np.hstack((np.squeeze(testParams[s][f].testPred[0]), np.squeeze(testParams[s][f].testPred[1]))))
        testPredicts = moving_average(testPredicts, 30)
        testPredictsAll.append(testPredicts)


    # Compute ROC curve and area the curve
    for testPredicts in testPredictsAll:
        fpr, tpr, thresholds = my_roc_curve(testLabels, testPredicts, pos_label=labelTypes["preictal"],
                                            neg_label=labelTypes["interictal"], refractory_period=900, rate_period=1800,
                                            point=5000)
        axes.plot(fpr, tpr)
    axes.set_ylabel("True Positive Rate (sensitivity)")
    axes.set_xlabel("False Positive Rate(per hour)")
    axes.set_ylim(0, 1.02)
    axes.set_xlim(-0.01, 1)

    fig.show()