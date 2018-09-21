import numpy as np
import pyedflib
import scipy.io

import math
from scipy.signal import welch
from scipy.stats import kurtosis, skew, moment
from functions.math_utils import my_trapz

bipolar_label=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',
               'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

unipolar_label = ['AF7', 'AF3', 'AF4', 'AF8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'T8', 'TP7', 'CP3', 'CPZ',
                  'CP4', 'TP8', 'PO7', 'PO3', 'PO4', 'PO8']

unipolar_reference = ['FP1-F7', 'FP1-F3', 'FP2-F4', 'FP2-F8', 'F7-T7', 'F3-C3', 'FZ-CZ', 'F4-C4', 'F8-T8', 'T7-FT9',
                      'FT10-T8', 'T7-P7', 'C3-P3', 'CZ-PZ', 'C4-P4', 'T8-P8', 'P7-O1', 'P3-O1', 'P4-O2', 'P8-O2']

idx_filter = []
idx_bands = []
idx_bands_len = []
idx_bands_truth = []


def checkSignalLabels(signal_labels):
    count = 0
    for labels in bipolar_label:
        if labels not in signal_labels:
            count = count + 1
    return count


def convertBipolar2UnipolarBasic(signal_labels, signalCell):

    output_data=[]

    for i in range(len(unipolar_label)):
        if unipolar_reference[i] in signal_labels:
            output_data.append([unipolar_label[i], signalCell[signal_labels.index(unipolar_reference[i])]])
        else:
            output_data.append([unipolar_label[i], np.zeros(len(signalCell[0]))])
    return output_data


def bandpower_setup(data, fs, band, filter_range=[]):

    freqs, psd = welch(data, fs, nperseg=len(data), scaling='density')

    idx_filter = np.ones(dtype=bool, shape=freqs.shape)
    for f in filter_range:
        low, high = np.asarray(f)
        # Find closest indices of band in frequency vector
        idx_min = np.argmax(freqs > low) - 1
        idx_max = np.argmax(freqs > high) - 1
        idx_filter[idx_min:idx_max] = False

    idx_bands = []
    idx_bands_len = [0]
    idx_bands_truth = []
    for b in band:
        low, high = np.asarray(b)
        # Find closest indices of band in frequency vector
        idx_min = np.argmax(freqs > low) - 1
        idx_max = np.argmax(freqs > high) - 1
        idx_band = np.zeros(dtype=bool, shape=freqs.shape)
        idx_band[idx_min:idx_max] = True
        idx_band = np.logical_and(idx_band, idx_filter)
        idx_bands.append(idx_band)
        idx_bands_len.append(idx_bands_len[-1] + idx_band.sum()-1)
        idx_bands_truth = np.hstack((idx_bands_truth, np.ones(dtype=bool, shape=idx_band.sum()-1), 0 ))
    idx_bands_len.pop(0)
    idx_bands_len.pop(-1)

    return idx_filter, idx_bands, idx_bands_len, np.asarray(idx_bands_truth, dtype=bool)

def bandpower(data, fs, band, filter_range=[], idx_filter=[], idx_bands=[], idx_bands_len=[], idx_bands_truth=[]):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.

    """
    # Compute the modified periodogram (Welch)

    freqs, psd = welch(data, fs, nperseg=len(data), scaling='density')

    if idx_filter == []:
        idx_filter = np.ones(dtype=bool, shape=freqs.shape)
        for f in filter_range:
            low, high = np.asarray(f)
            # Find closest indices of band in frequency vector
            idx_min = np.argmax(freqs > low)
            idx_max = np.argmax(freqs > high)
            idx_filter[idx_min:idx_max] = False

    if idx_bands == []:
        idx_bands = []
        idx_bands_len = [0]
        idx_bands_truth = []
        for b in band:
            low, high = np.asarray(b)
            # Find closest indices of band in frequency vector
            idx_min = np.argmax(freqs > low) - 1
            idx_max = np.argmax(freqs > high) - 1
            idx_band = np.zeros(dtype=bool, shape=freqs.shape)
            idx_band[idx_min:idx_max] = True
            idx_band = np.logical_and(idx_band, idx_filter)
            idx_bands.append(idx_band)
            idx_bands_len.append(idx_bands_len[-1] + idx_band.sum() - 1)
            idx_bands_truth = np.hstack((idx_bands_truth, np.ones(dtype=bool, shape=idx_band.sum() - 1), 0))
        idx_bands_len.pop(0)
        idx_bands_len.pop(-1)
        idx_bands_truth = np.asarray(idx_bands_truth, dtype=bool)

    bps_all = my_trapz(psd[idx_filter], freqs[idx_filter])
    bps_split = np.split(bps_all[idx_bands_truth], idx_bands_len)
    bps = [sum(bp) for bp in bps_split]
    # bps = []
    # for idx_band in idx_bands:
    #     # Integral approximation of the spectrum using Simpson's rule.
    #     #bp = simps(psd[idx_band], freqs[idx_band])
    #     bp = trapz(psd[idx_band], freqs[idx_band])
    #     bps.append(bp)

    return bps


def calc_spectral_band_power(signal, fs, setup=False):
    filt_range = [[0, 0.5], [57, 63], [117, 123]]
    filt_range = np.vstack((filt_range, [[15.7, 16.3], [31.7, 32.3], [47.7, 48.3], [63.7, 64.3], [75.7, 76.3], [79.7, 80.3], [95.7, 96.3], [111.7, 112.3], [127.7, 128]]))
    delta = [0, 4]
    teta = [4, 8]
    alpha = [8, 13]
    beta = [13, 30]
    gama1 = [30, 50]
    gama2 = [50, 75]
    gama3 = [75, 100]
    gama4 = [100, 128]
    freqs = [delta, teta, alpha, beta, gama1, gama2, gama3, gama4]

    if setup:
        global idx_filter
        global idx_bands
        global idx_bands_len
        global idx_bands_truth
        idx_filter, idx_bands, idx_bands_len, idx_bands_truth = bandpower_setup(signal, fs, freqs, filt_range)
        return None
    else:
        bp = bandpower(signal, fs, freqs, filt_range, idx_filter, idx_bands, idx_bands_len, idx_bands_truth)
        return bp


def calc_statistical_moments(signal):
    m = np.mean(signal)
    v = np.var(signal)
    m3 = moment(signal, moment=3)
    m4 = moment(signal, moment=4)
    if v > 0:
        sk = m3/(math.sqrt(math.pow(v, 3)))
        kurt = (m4 / math.pow(v, 2)) - 3
    else:
        sk = 0
        kurt = 0
    return m, v, sk, kurt


def calc_hjorth(signal):

    signald = np.diff(signal)
    signaldd = np.diff(signald)

    var_signal = np.var(signal)
    var_signald = np.var(signald)
    var_signaldd = np.var(signaldd)

    if var_signal > 0:
        mobility = math.sqrt(var_signald/var_signal)
        if mobility:
            complexity = math.sqrt(var_signaldd/var_signald) / mobility
        else:
            complexity = 0
    else:
        mobility = 0
        complexity = 0
    return mobility, complexity

def extractFeatures(signal, fs, window_length, overlap):
    features = []
    step_length = np.int(window_length * fs * (1 - overlap))

    # bandpower setup
    calc_spectral_band_power(signal[0][1][0:window_length * fs], fs, setup=True)
    signal_len = len(signal[0][1])
    signal_len = signal_len - (signal_len % step_length)
    for w in range(0, signal_len - step_length, step_length):
        window_features = []
        for s in signal:
            data = s[1][w:w+window_length*fs]
            bp = calc_spectral_band_power(data, fs)
            sm = calc_statistical_moments(data)
            hj = calc_hjorth(data)
            window_features.append(np.hstack([bp, sm, hj]))
        features.append(window_features)
    return features
