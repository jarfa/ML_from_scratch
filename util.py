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

def logloss(observed, predicted, trim=1e-9):
    # keep loss from being infinite
    predicted = np.clip(predicted, trim, 1.0 - trim)
    return -np.mean(
        observed * np.log(predicted) + 
        (1. - observed) * np.log(1. - predicted)
    )

def normLL(raw_logloss, baserate):
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
        raise Exception("Data and targets do not have the same number of rows.")
    shuffle_ix = np.random.permutation(len(targets))
    return data[shuffle_ix,:], targets[shuffle_ix]

def report(model_name, train_time, pred_time, nll):
    line = "\t".join([
        "{mod: <25}".format(mod=model_name or ""),
        "norm LL: {nll:.3f}".format(nll=nll),
        "train time: {train:.3f}s".format(train=train_time),
        "pred time: {pred:.3f}s".format(pred=pred_time),
    ])
    print(line)
