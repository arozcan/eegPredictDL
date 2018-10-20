import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functions.math_utils import moving_average
from sklearn.metrics import roc_curve, auc
from functions.ranking import my_roc_curve
from functions.globals import labelTypes, dbFeatGroups
from functions.utils import find_ranges
import itertools
from scipy.spatial.distance import euclidean


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def find_cutoff_idx(fpr, tpr):
    r=[]
    for f, t in zip(fpr, tpr):
        r.append(np.sqrt(f ** 2 + (1-t) ** 2))
    r.reverse()
    return len(fpr)-np.argmin(r)-1

def get_plot_label_values(result, plotLabels):

    value = ""
    for label in plotLabels:
        for groups in dbFeatGroups:
            if label in result.get(groups).__dict__.keys():
                if value:
                    value += ", "
                val = result.get(groups).__dict__.get(label)
                if label == "subject":
                    value += label + ": " + str(val+1)
                    print(label + ": " + str(val+1))
                elif isinstance(val, basestring):
                    value += str(val)
                elif isinstance(val, np.ndarray) or isinstance(val, list):
                    value += label + ": " + str(val).replace(" ", "")
                elif isinstance(val, bool):
                    if val:
                        value += label + ": True"
                    else:
                        value += label + ": False"
                elif isinstance(val, int):
                    value += label + ": " +str(int(val))
                elif isinstance(val, dict):
                    if val["active"]:
                        value += label + ": True"
                    else:
                        value += label + ": False"
                elif isinstance(val, float):
                    value += label + ": " + str(val)
    return value


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


def plot_subject_prediction_train(testParams, foldPairs, labels):

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


def plot_subject_scores(testParams, foldPairs, labels, featureParams, timingParams):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('ROC Analysis', fontweight='bold')

    stepLen = (featureParams.windowLength * (1 - featureParams.overlap))
    refractoryLen = timingParams.refractoryLen * 60
    refractoryPeriod = int(refractoryLen / stepLen)

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
                                            neg_label=labelTypes["interictal"], refractory_period=refractoryPeriod,
                                            rate_period=60*60/stepLen, point=5000)
        axes.plot(fpr, tpr)
    axes.set_ylabel("True Positive Rate (sensitivity)")
    axes.set_xlabel("False Positive Rate(per hour)")
    axes.set_ylim(0, 1.02)
    axes.set_xlim(-0.01, 1)

    fig.show()


def plot_subject_roc(result, plotTitle=None):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('ROC Analysis', fontweight='bold')
    colors = cm.gist_ncar(np.linspace(0, 1, 8))

    featureParams = result['featureParams']
    timingParams = result['timingParams']
    labels = result['dataset'].labels
    foldPairs = result["dataset"].fold_pairs
    testParams = result["trainResult"].testParams

    stepLen = (featureParams.windowLength * (1 - featureParams.overlap))
    refractoryLen = timingParams.refractoryLen * 60
    refractoryPeriod = int(refractoryLen / stepLen)


    testLabels = np.full(len(labels), fill_value=-1, dtype=int)

    for s in range(len(testParams)):
        testLabels[foldPairs[s][0][2]] = np.squeeze(
            np.hstack((np.full(len(testParams[s][0].testPred[0]),fill_value=labelTypes["interictal"]),
                       np.full(len(testParams[s][0].testPred[1]),fill_value=labelTypes["preictal"]))))

    testPredictsMean = np.zeros(len(labels), dtype=float)
    testPredictsAll = []
    for f in range(len(testParams[0])):
        testPredicts = np.zeros(len(labels), dtype=float)
        for s in range(len(testParams)):
            testPredicts[foldPairs[s][f][2]] += np.squeeze(
                np.hstack((np.squeeze(testParams[s][f].testPred[0]), np.squeeze(testParams[s][f].testPred[1]))))
        testPredicts = moving_average(testPredicts, 30)
        testPredictsMean += testPredicts / len(testParams[0])
        testPredictsAll.append(testPredicts)



    plotLabel = "fold-{}"
    for testPredicts, color, idx in zip(testPredictsAll, colors, range(len(testPredictsAll))):
        fpr, tpr, thresholds = my_roc_curve(testLabels, testPredicts, pos_label=labelTypes["preictal"],
                                            neg_label=labelTypes["interictal"], refractory_period=refractoryPeriod,
                                            rate_period=60*60/stepLen, point=2000)
        cutoff = np.argmax(tpr)
        axes.plot(fpr, tpr, label=plotLabel.format(idx+1), color=color)
        axes.plot(fpr[cutoff], tpr[cutoff], 'o', mfc="None", color=color, label="({},{})".format(round(fpr[cutoff], 2),
                                                                                                 round(tpr[cutoff]), 2))
    axes.set_ylabel("True Positive Rate (sensitivity)")
    axes.set_xlabel("False Positive Rate (per hour)")
    axes.set_ylim(0, 1.02)
    axes.set_xlim(-0.01, 1)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(flip(handles, 2), flip(labels, 2), ncol=2, loc=4, handletextpad=0.15, columnspacing=0.2)

    if plotTitle:
        axes.set_title(get_plot_label_values(result, plotTitle),loc="center")

    fig.show()


def plot_subject_prediction(result, plotTitle=None):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Prediction Results', fontweight='bold')
    colors = cm.gist_ncar(np.linspace(0, 1, 8))
    tickPeriod = 60*60

    featureParams = result['featureParams']
    timingParams = result['timingParams']
    labels = result['dataset'].labels
    foldPairs = result["dataset"].fold_pairs
    testParams = result["trainResult"].testParams

    stepLen = (featureParams.windowLength * (1 - featureParams.overlap))

    testLabels = np.full(len(labels), fill_value=-1, dtype=int)

    for s in range(len(testParams)):
        testLabels[foldPairs[s][0][2]] = np.squeeze(
            np.hstack((np.full(len(testParams[s][0].testPred[0]), fill_value=labelTypes["interictal"]),
                       np.full(len(testParams[s][0].testPred[1]), fill_value=labelTypes["preictal"]))))

    testPredictsMean = np.zeros(len(labels), dtype=float)
    for f in range(len(testParams[0])):
        testPredicts = np.zeros(len(labels), dtype=float)
        for s in range(len(testParams)):
            testPredicts[foldPairs[s][f][2]] += np.squeeze(
                np.hstack((np.squeeze(testParams[s][f].testPred[0]), np.squeeze(testParams[s][f].testPred[1]))))
        testPredicts = moving_average(testPredicts, 30)
        testPredictsMean += testPredicts / len(testParams[0])

    axes.plot(testPredictsMean)
    axes.set_ylabel("Prediction")
    axes.set_xlabel("Time (hour)")
    ticks = range(0, len(testPredictsMean), int(tickPeriod/stepLen))
    axes.set_xticks(ticks, minor=False)
    axes.set_xticklabels(range(len(ticks)), fontdict=None, minor=False)
    axes.xaxis.grid(which='major', alpha=0.7, linestyle=':')

    pos_idx = find_ranges(testLabels, val=1)
    seizure_idx = np.transpose(pos_idx)[1] + int(timingParams.sphLen * 60 / stepLen)
    for sei, idx in zip(seizure_idx, range(len(seizure_idx))):
        if idx==0:
            axes.axvline(x=sei, linewidth=2, color='r', label='seizure onsets')
        else:
            axes.axvline(x=sei, linewidth=2, color='r')
    axes.set_xlim(0, len(testPredictsMean))
    axes.set_ylim(0, 1.05)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, ncol=1, loc=0, handletextpad=0.5, columnspacing=0.2)

    if plotTitle:
        axes.set_title(get_plot_label_values(result, plotTitle), loc="center")

    fig.show()


def compare_subject_roc(results, plotLabels=None, plotTitle=None):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('ROC Analysis', fontweight='bold')
    colors = cm.gist_ncar(np.linspace(0, 1, 8))

    for result, color in zip(results, colors):
        featureParams = result['featureParams']
        timingParams = result['timingParams']
        labels = result['dataset'].labels
        foldPairs = result["dataset"].fold_pairs
        testParams = result["trainResult"].testParams

        stepLen = (featureParams.windowLength * (1 - featureParams.overlap))
        refractoryLen = timingParams.refractoryLen * 60
        refractoryPeriod = int(refractoryLen / stepLen)


        testLabels = np.full(len(labels), fill_value=-1, dtype=int)

        for s in range(len(testParams)):
            testLabels[foldPairs[s][0][2]] = np.squeeze(
                np.hstack((np.full(len(testParams[s][0].testPred[0]),fill_value=labelTypes["interictal"]),
                           np.full(len(testParams[s][0].testPred[1]),fill_value=labelTypes["preictal"]))))

        testPredictsMean = np.zeros(len(labels), dtype=float)
        for f in range(len(testParams[0])):
            testPredicts = np.zeros(len(labels), dtype=float)
            for s in range(len(testParams)):
                testPredicts[foldPairs[s][f][2]] += np.squeeze(
                    np.hstack((np.squeeze(testParams[s][f].testPred[0]), np.squeeze(testParams[s][f].testPred[1]))))
            testPredicts = moving_average(testPredicts, 30)
            testPredictsMean += testPredicts / len(testParams[0])

        fpr, tpr, thresholds = my_roc_curve(testLabels, testPredictsMean, pos_label=labelTypes["preictal"],
                                            neg_label=labelTypes["interictal"], refractory_period=refractoryPeriod,
                                            rate_period=60*60/stepLen, point=6000)
        cutoff=find_cutoff_idx(fpr, tpr)

        if plotLabels:
            plotLabel=get_plot_label_values(result, plotLabels)

        axes.plot(fpr, tpr, label=plotLabel, color=color)
        axes.plot(fpr[cutoff], tpr[cutoff], 'o', mfc="None", color=color, label="({},{})".format(round(fpr[cutoff], 2),
                                                                                                 round(tpr[cutoff], 2)))
    axes.set_ylabel("True Positive Rate (sensitivity)")
    axes.set_xlabel("False Positive Rate (per hour)")
    axes.set_ylim(0, 1.02)
    axes.set_xlim(-0.01, 1)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(flip(handles, 2), flip(labels, 2), ncol=2, loc=4, handletextpad=0.15, columnspacing=0.2)

    if plotTitle:
        axes.set_title(get_plot_label_values(result, plotTitle), loc="center")

    fig.show()


def compare_subject_prediction(results, plotLabels=None, plotTitle=None):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Prediction Results', fontweight='bold')
    colors = cm.gist_ncar(np.linspace(0, 1, 8))
    tickPeriod = 60*60

    for result, color in zip(results, colors):
        featureParams = result['featureParams']
        timingParams = result['timingParams']
        labels = result['dataset'].labels
        foldPairs = result["dataset"].fold_pairs
        testParams = result["trainResult"].testParams

        stepLen = (featureParams.windowLength * (1 - featureParams.overlap))

        testLabels = np.full(len(labels), fill_value=-1, dtype=int)
        intHour = 0

        for s in range(len(testParams)):
            intHour += len(testParams[s][0].testPred[0])
            testLabels[foldPairs[s][0][2]] = np.squeeze(
                np.hstack((np.full(len(testParams[s][0].testPred[0]), fill_value=labelTypes["interictal"]),
                           np.full(len(testParams[s][0].testPred[1]), fill_value=labelTypes["preictal"]))))
        intHour = intHour * stepLen / (60*60)

        testPredictsMean = np.zeros(len(labels), dtype=float)
        for f in range(len(testParams[0])):
            testPredicts = np.zeros(len(labels), dtype=float)
            for s in range(len(testParams)):
                testPredicts[foldPairs[s][f][2]] += np.squeeze(
                    np.hstack((np.squeeze(testParams[s][f].testPred[0]), np.squeeze(testParams[s][f].testPred[1]))))
            testPredicts = moving_average(testPredicts, 30)
            testPredictsMean += testPredicts / len(testParams[0])

        if plotLabels:
            plotLabel = get_plot_label_values(result, plotLabels)

        axes.plot(testPredictsMean, label=plotLabel, color=color)
    axes.set_ylabel("Prediction")
    axes.set_xlabel("Time (hour)")
    ticks = range(0, len(testPredictsMean), int(tickPeriod/stepLen))
    axes.set_xticks(ticks, minor=False)
    axes.set_xticklabels(range(len(ticks)), fontdict=None, minor=False)
    axes.xaxis.grid(which='major', alpha=0.7, linestyle=':')

    pos_idx = find_ranges(testLabels, val=1)
    seizure_idx = np.transpose(pos_idx)[1] + int(timingParams.sphLen * 60 / stepLen)
    for sei, idx in zip(seizure_idx, range(len(seizure_idx))):
        if idx==0:
            axes.axvline(x=sei, linewidth=2, color='r', label='seizure onsets')
        else:
            axes.axvline(x=sei, linewidth=2, color='r')
    axes.set_xlim(0, len(testPredictsMean))
    axes.set_ylim(0, 1.05)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, ncol=1, loc=0, handletextpad=0.5, columnspacing=0.2)

    if plotTitle:
        axes.set_title(get_plot_label_values(result, plotTitle), loc="center")

    print("\tInterictal Hour {}".format(intHour))

    fig.show()