from __future__ import absolute_import, print_function, division, unicode_literals
import warnings
import numpy as np


def normalize(data, mean=None, sd=None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if sd is None:
        sd = np.std(data, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalized = np.divide(data - mean, sd)
    normalized[np.isnan(normalized)] = 0.0
    normalized[np.isinf(normalized)] = 0.0
    return normalized


def roc_curve(observed, predicted, presorted=False):
    if not presorted:
         # We don't need the actual scalar predictions, just the
         # observed events sorted by predictions (descending).
        sort_ix = predicted.argsort()[::-1]
        observed = observed[sort_ix]
    N_pos = sum(observed)
    N_neg = len(observed) - N_pos
    tp = fp = 0.0
    true_pos_rate = [] #a.k.a. recall (# true pos / # observed pos)
    false_pos_rate = [] #a.k.a. 1 - sensitivity (# false pos / # observed neg)
    for obs in observed:
        # obs will only be either 1 or 0
        tp += obs
        fp += 1 - obs
        true_pos_rate.append(tp / N_pos)
        false_pos_rate.append(fp / N_neg)

    return true_pos_rate, false_pos_rate


def roc_auc(observed, predicted, presorted=False):
    if set(observed) != set([0, 1]):
        raise ValueError("Observed data must be binary (1,0)")
    N_tot = len(observed)
    if N_tot != len(predicted):
        raise ValueError("Arrays must be of equal length")
    # I'm separating out these functions to ease testing (and to have my output
    # be more closely comparable to sklearn's implementation)
    tpr, fpr = roc_curve(observed, predicted, presorted=presorted)
    # start with the area of the first trapezoid under the curve. This looks
    # weird but the alterantive is less clean.
    auc = fpr[0] * tpr[0] / 2.0
    for i in range(N_tot)[1:]:
        # adding the area of each additional trapezoid
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0

    return auc


def logloss(observed, predicted, trim=1e-9):
    # keep loss from being infinite
    predicted = np.clip(predicted, trim, 1.0 - trim)
    return -np.mean(
        observed * np.log(predicted) + 
        (1. - observed) * np.log(1. - predicted)
    )


def normLL(raw_logloss, baserate):
    # compute what logloss would be if you always predicted the baserate
    ll_br = -(baserate * np.log(baserate) + (1 - baserate) * np.log(1 - baserate))
    return 1. - (raw_logloss / ll_br)


def logit(prob):
    return np.log(prob / (1.0 - prob))


def ilogit(log_odds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1. / (1. + np.exp(-log_odds))


def shuffle_rows(data, targets):
    if not data.shape[0] == len(targets):
        raise ValueError("Data and targets do not have the same number of rows.")
    shuffle_ix = np.random.permutation(len(targets))
    return data[shuffle_ix,:], targets[shuffle_ix]


def report(model_name, train_time, pred_time, **metrics):
    """Reports as many metrics as requested"""
    report_items = ["{mod: <30}".format(mod=model_name or "")]

    for mname, mval in metrics.items():
        report_items.append("{0}: {1:.3f}".format(mname, mval))

    report_items += [
            "train time: {train:.3f}s".format(train=train_time),
            "pred time: {pred:.3f}s".format(pred=pred_time),
    ]
    print("    ".join(report_items))
