import numpy as np
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.fixes import array_equal
from sklearn.utils.extmath import stable_cumsum
from functions.utils import find_ranges

def my_binary_clf_curve(y_true, y_score, pos_label=1, neg_label=0, refractory_period=1, point=1000):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int or str, default=None
        The label of the positive class

    refractory_period: refractory period of prediction

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    check_consistent_length(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_desc = y_score[desc_score_indices]

    pos_idx=find_ranges(y_true, val=pos_label)
    neg_idx=find_ranges(y_true, val=neg_label)


    min_pos_score = np.min([np.max(y_score[idx[0]:idx[1]]) for idx in pos_idx])
    max_pos_scores = [np.max(y_score[idx[0]:idx[1]]) for idx in pos_idx]

    imp_tresh = [0,1]
    imp_tresh= np.hstack((imp_tresh, max_pos_scores, np.add(max_pos_scores, 0.00000001),
                          np.add(max_pos_scores, -0.00000001)))
    tresholds = np.sort(np.unique(imp_tresh))[::-1]
    # tresholds = np.arange(min_pos_score, 1, float(1)/point)
    # tresholds = np.sort(np.unique(np.hstack((tresholds, imp_tresh))))[::-1]


    tps = []
    fps = []
    for tr in tresholds:
        tps.append(0)
        fps.append(0)
        if np.count_nonzero(y_score >= tr):
            i = np.min(np.squeeze(np.where(y_score >= tr)))
            while i < len(y_true):
                if y_score[i] >= tr:
                    if y_true[i] == pos_label:
                        tps[-1] += 1
                        if np.count_nonzero(np.transpose(pos_idx)[1]>i):
                            i = pos_idx[np.squeeze(np.where(np.transpose(pos_idx)[1]>i)).flat[0]][1] + 1
                        else:
                            i += refractory_period
                    elif y_true[i] == neg_label:
                        fps[-1] += 1
                        if pos_label in y_true[i:i+refractory_period]:
                            if np.count_nonzero(y_true[i:] == pos_label):
                                i += np.min(np.squeeze(np.where(y_true[i:] == pos_label)))
                            else:
                                i += refractory_period
                        else:
                            i +=refractory_period
                    else:
                        if np.count_nonzero(y_true[i:] == neg_label) or np.count_nonzero(y_true[i:] == pos_label):
                            i += np.min(np.hstack((np.squeeze(np.where(y_true[i:] == neg_label)),
                                                   (np.squeeze(np.where(y_true[i:] == pos_label))))))
                        else:
                            i += refractory_period
                else:
                    if i < len(y_true):
                        if np.count_nonzero(y_score[i:] >= tr):
                            i += np.min(np.squeeze(np.where(y_score[i:] >= tr)))
                        else:
                            break


    return np.asarray(fps, dtype=float), np.asarray(tps, dtype=float), np.asarray(tresholds)

def my_roc_curve(y_true, y_score, pos_label=1, neg_label=0, refractory_period=1, rate_period=1, point=1000):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

        .. versionadded:: 0.17
           parameter *drop_intermediate*.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    See also
    --------
    roc_auc_score : Compute Area Under the Curve (AUC) from prediction scores

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([ 0. ,  0.5,  0.5,  1. ])
    >>> tpr
    array([ 0.5,  0.5,  1. ,  1. ])
    >>> thresholds
    array([ 0.8 ,  0.4 ,  0.35,  0.1 ])

    """
    fps, tps, thresholds = my_binary_clf_curve(y_true, y_score, pos_label=pos_label, neg_label=neg_label,
                                               refractory_period=refractory_period, point=point)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        p_count = np.count_nonzero(y_true == neg_label)
        div = p_count/rate_period
        fpr = fps / div

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    fps = np.r_[0, fps]
    tps = np.r_[0, tps]
    thresholds = np.r_[1, thresholds]

    return fpr, tpr, fps, tps, thresholds