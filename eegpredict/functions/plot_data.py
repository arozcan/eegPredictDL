import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import datetime
from functions.math_utils import moving_average
from sklearn.metrics import roc_curve, auc
from functions.ranking import my_roc_curve
from functions.globals import labelTypes, dbFeatGroups, excelWriter
from functions.utils import find_ranges, merge_dicts, in_notebook, dashed_line
from functions.io_utils import load_results, get_from_test_db
import itertools
import pickle
import os
from functions import globals
from IPython.display import Markdown, display

pd.set_option('display.width', 320)


def printmd(string, color=None, center=False, h=None):
    str = "<span style='color:{}'>{}</span>".format(color, string)
    if center:
        str = "<center>{}</center>".format(str)
    if h:
        str = "<h{}>{}</h{}>".format(h,str,h)
    display(Markdown(str))


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
                    #print(label + ": " + str(val+1))
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
        fpr, tpr, fps, tps, thresholds = my_roc_curve(testLabels, testPredicts, pos_label=labelTypes["preictal"],
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
    #colors = cm.gist_ncar(np.linspace(0, 1, 4))
    colors =cm.tab20(np.linspace(0, 1, 20))

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
    testPredictsAll.append(testPredictsMean)

    plotLabel = np.hstack((["fold-{}".format(x+1) for x in range(len(testParams[0]))], "mean"))
    for testPredicts, color, idx in zip(testPredictsAll, colors, range(len(testPredictsAll))):
        fpr, tpr, fps, tps, thresholds = my_roc_curve(testLabels, testPredicts, pos_label=labelTypes["preictal"],
                                            neg_label=labelTypes["interictal"], refractory_period=refractoryPeriod,
                                            rate_period=60*60/stepLen, point=20)
        cutoff = find_cutoff_idx(fpr, tpr)
        axes.plot(fpr, tpr, label=plotLabel[idx], color=color)
        axes.plot(fpr[cutoff], tpr[cutoff], 'o', mfc="None", color=color, label="({},{})".format(round(fpr[cutoff], 2),
                                                                                                 round(tpr[cutoff], 2)))
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
    #colors = cm.gist_ncar(np.linspace(0, 1, 4))
    colors =cm.tab20(np.linspace(0, 1, 20))
    tickPeriod = 60*60

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

    print("\tinterictal Hour:{}\tseizure Count:{}".format(intHour, len(seizure_idx)))

    fig.show()


def compare_subject_roc(results, plotLabels=None, plotTitle=None):

    fig, axes = plt.subplots(1, 1)
    fig.suptitle('ROC Analysis', fontweight='bold')
    #colors = cm.gist_ncar(np.linspace(0, 1, 4))
    colors =cm.tab20(np.linspace(0, 1, 20))

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

        fpr, tpr, fps, tps, thresholds = my_roc_curve(testLabels, testPredictsMean, pos_label=labelTypes["preictal"],
                                            neg_label=labelTypes["interictal"], refractory_period=refractoryPeriod,
                                            rate_period=60*60/stepLen, point=20)
        cutoff=find_cutoff_idx(fpr, tpr)

        if plotLabels:
            plotLabel=get_plot_label_values(result, plotLabels)

        axes.plot(fpr, tpr, label=plotLabel, color=color)
        axes.plot(fpr[cutoff], tpr[cutoff], 'o', mfc="None", color=color, label="({},{})".format(round(fpr[cutoff], 2),
                                                                                                 round(tpr[cutoff], 2)))
        print("\t"+plotLabel + "\tfpr:{:2.4f}\ttpr:{:2.4f}\tfps:{:2d}\ttps:{:2d}".format(fpr[cutoff],tpr[cutoff],int(fps[cutoff]),int(tps[cutoff])))

    axes.set_ylabel("Sensitivity")
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
    #colors = cm.gist_ncar(np.linspace(0, 1, 4))
    colors =cm.tab20(np.linspace(0, 1, 20))
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
    axes.set_ylabel("Prediction Value")
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

    print("\tinterictal Hour:{}\tseizure Count:{}".format(intHour,len(seizure_idx)))

    fig.show()


def gen_subject_roc(test_id):

    result=load_results(test_id)

    featureParams = result['featureParams']
    timingParams = result['timingParams']
    labels = result['dataset'].labels
    foldPairs = result["dataset"].fold_pairs
    testParams = result["trainResult"].testParams

    stepLen = (featureParams.windowLength * (1 - featureParams.overlap))
    refractoryLen = timingParams.refractoryLen * 60
    refractoryPeriod = int(refractoryLen / stepLen)


    testLabels = np.full(len(labels), fill_value=-1, dtype=int)
    intHour = 0

    for s in range(len(testParams)):
        intHour += len(testParams[s][0].testPred[0])
        testLabels[foldPairs[s][0][2]] = np.squeeze(
            np.hstack((np.full(len(testParams[s][0].testPred[0]),fill_value=labelTypes["interictal"]),
                       np.full(len(testParams[s][0].testPred[1]),fill_value=labelTypes["preictal"]))))
    intHour = intHour * stepLen / (60 * 60)

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
    testPredictsAll.append(testPredictsMean)

    fpr_all = []
    tpr_all = []
    fps_all = []
    tps_all = []
    thresholds_all = []
    cutoff_all = []
    for testPredicts, idx in zip(testPredictsAll, range(len(testPredictsAll))):
        fpr, tpr, fps, tps, thresholds = my_roc_curve(testLabels, testPredicts, pos_label=labelTypes["preictal"],
                                            neg_label=labelTypes["interictal"], refractory_period=refractoryPeriod,
                                            rate_period=60*60/stepLen, point=20)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        fps_all.append(fps)
        tps_all.append(tps)
        thresholds_all.append(thresholds)
        cutoff_all.append(find_cutoff_idx(fpr, tpr))

    results = {"fpr": fpr_all,
               "tpr": tpr_all,
               "fps": fps_all,
               "tps": tps_all,
               "thresholds": thresholds_all,
               "cutoff": cutoff_all,
               "intHour": intHour,
               "sCount": int(tps_all[-1][-1])}

    # save results and params
    test_name = globals.file_test + "test_roc_" + str(test_id) + ".mat"
    pickle_out = open(test_name, "wb")
    pickle.dump(results, pickle_out)
    pickle_out.close()


def get_subject_roc(test_id):

    test_name = globals.file_test + "test_roc_" + str(test_id) + ".mat"
    if not os.path.isfile(test_name):
        gen_subject_roc(test_id)
    pickle_in = open(test_name, "rb")
    results = pickle.load(pickle_in)
    pickle_in.close()

    return results


def find_matched_elems(compareList, params):
    params_ = params.copy()
    for param in params_:
        newParam = param.replace(' ', '').split(',')
        if len(newParam)>1:
            newDict = merge_dicts([{n: params[param][i]} for n, i in zip(newParam, range(len(newParam)))])
            params.pop(param)
            params = merge_dicts([params, newDict])
    selectedId = [set([i for c, i in zip(compareList, range(len(compareList)))
                        if c.__getattribute__(k) == params.get(k)]) for k in params.keys()]
    selectedId = selectedId[0].intersection(*selectedId[1:])
    return [compareList[i] for i in selectedId]


def print_selected_compare(selectedCompare, results=["fpr","tpr","sCount","intHour"], merge=["sCount","intHour"]):

    compareFileName = globals.file_comparison+"comparison_"+','.join([par for par in selectedCompare[0].compareParams])
    writer = excelWriter(compareFileName)

    for compareList, cmpidx in zip(selectedCompare, range(len(selectedCompare))):
        selectedParams = ', '.join([str(params) + "=" + str(compareList.selectedParams.get(params))
                                                   for params in compareList.selectedParams])
        header = "Selected Params:\t" + selectedParams

        sortedIdx = np.argsort(compareList.compareArray.shape)
        compareListParams = [compareList.compareParams[i] for i in sortedIdx]

        dictArray=[]
        for comp in compareList.compareArray.ravel():
            compareParams = merge_dicts([{par: comp.__getattribute__(par)} for par in compareListParams])
            resultParams=merge_dicts([{par: comp.get_cutoff_mean_values(par)} for par in results])
            dictArray.append(merge_dicts([compareParams, resultParams]))

        d1 = pd.DataFrame(dictArray)
        d1 = d1.set_index(compareListParams)
        d1 = d1.unstack(compareListParams[:-1])

        for m in merge:
            temp = d1[m].mean(axis=1)
            d1=d1.drop([m], axis=1)
            d1 = d1.rename(columns={m: 'dropped'})
            d1[m] = temp

        d1.loc['Total'] = 0
        if 'intHour' in results:
            if 'fpr' in results:
                temp = []
                for c in d1['fpr'].columns:
                    temp.append(sum(d1['fpr'][c] * d1['intHour'])/sum(d1['intHour']))
                d1.loc['Total']['fpr'] = temp
            d1.loc['Total']['intHour'] = sum(d1['intHour'])

        if 'sCount' in results:
            if 'tpr' in results:
                temp = []
                for c in d1['tpr'].columns:
                    temp.append(sum(d1['tpr'][c] * d1['sCount'])/sum(d1['sCount']))
                d1.loc['Total']['tpr'] = temp
            d1.loc['Total']['sCount'] = sum(d1['sCount'])


        if in_notebook():
            printmd(header, center=True)
            display(d1.style)
        else:
            print(header)
            print(d1)
            writer.write(d1,title=selectedParams)





def compare_various_result(compareParams, validParams, selectSet):

    compareList = []
    for compareParam in itertools.product(*compareParams):
        compareParam = merge_dicts(compareParam)
        results = []
        ids = get_from_test_db(feats=compareParam)
        for id in ids:
            results.append(get_subject_roc(id[0]))
        compareList.append(globals.CompareClass(validParams, compareParam, results))

    validParamList = {}
    for valid in validParams:
        for compareParam in compareParams:
            for cp in compareParam:
                vp = valid.replace(' ', '').split(',')
                if vp[0] in cp:
                    if validParamList.has_key(valid):
                        if len(vp) > 1:
                            value = [cp.get(v) for v in vp]
                            validParamList.get(valid).append(value)
                        else:
                            validParamList.get(valid).append(cp.get(valid))
                    else:
                        if len(vp) > 1:
                            value = [cp.get(v) for v in vp]
                            validParamList.update({valid: [value]})
                        else:
                            validParamList.update({valid: [cp.get(valid)]})

    # select results
    selectedCompareList = []
    for sel in selectSet:
        selectedCompare = []
        for selectedParams in sel:
            vp = np.hstack([vp.replace(' ', '').split(',') for vp in validParams])
            matchedList = find_matched_elems(compareList, selectedParams)
            compareParams = list(np.setdiff1d(validParams, ",".join(selectedParams.keys())))
            compareArray = np.empty([len(validParamList[other]) for other in compareParams], dtype=list)
            for params, idx in zip(itertools.product(*[validParamList[other] for other in compareParams]),
                                         itertools.product(*[range(len(validParamList[other])) for other in compareParams])):
                compareParamDict = merge_dicts([{v: c} for c, v in zip(params, compareParams)])
                compareArray[idx] = find_matched_elems(matchedList, compareParamDict)[0]
            selectedCompare.append(globals.SelectedCompareClass(selectedParams, compareParams, compareArray))
        selectedCompareList.append(selectedCompare)

    # print compares
    for selectedCompare in selectedCompareList:
        print_selected_compare(selectedCompare)
