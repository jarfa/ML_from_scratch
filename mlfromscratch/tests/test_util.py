import unittest
import numpy as np
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from mlfromscratch.util import *

def rounded_list(array, digits=6):
    """
    Because the unittest module won't do almost equal
    comparisons of numpy arrays or even lists :(
    """
    return list(np.round(array, digits))


def data_to_norm(N):
    rng = np.random.RandomState(5)
    return np.array([
            rng.choice([0, 0, 1, 15], size=N), #multinomial
            rng.rand(N), #uniform (0,1)
            5 * rng.randn(N) - 3, #Normal(-3, 5)
        ]).T


class Test_normalize(unittest.TestCase):
    def test_norm_no_params(self):
        data = data_to_norm(20)
        normed = normalize(data)
        # comparing numpy arrays with the unittest module is a bit ugly
        self.assertListEqual(
            rounded_list(np.mean(normed, axis=0)),
            [0.0] * 3
        )
        self.assertListEqual(
            rounded_list(np.std(normed, axis=0)),
            [1.0] * 3
        )

    def test_norm_defined_params(self):
        data = data_to_norm(20)
        means = np.mean(data, axis=0)
        stdevs = np.std(data, axis=0)
        normed = normalize(data, mean=1, sd=3)
        self.assertListEqual(
            rounded_list(np.mean(normed, axis=0)),
            rounded_list((means - 1.) / 3.)
        )
        self.assertListEqual(
            rounded_list(np.std(normed, axis=0)),
            rounded_list(stdevs / 3, 6)
        )


class Test_roc(unittest.TestCase):
    def test_roc_curve(self):
        obs = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        pred = np.arange(len(obs))
        true_pos, false_pos = roc_curve(obs, pred)
        self.assertListEqual(
            true_pos,
            [0.25, 0.25, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0]
        )
        self.assertListEqual(
            false_pos,
            [0.0, 0.25, 0.25, 0.25, 0.5, 0.75, 0.75, 1.0]
        )

    def test_roc_auc_final(self):
        rng = np.random.RandomState(5)
        N = 10**4
        pred = np.arange(0.0, 1.0, step=1. / N)
        obs = np.array(rng.rand(N) < pred, dtype=int)
        self.assertAlmostEqual(
            roc_auc(obs, pred),
            sklearn_roc_auc_score(obs, pred)
        )

    def test_roc_auc_binary_date(self):
        with self.assertRaises(ValueError):
            roc_auc(
                np.array([0, 1, 3, 1]),
                np.ones(4) * 0.1
        )

    def test_roc_auc_mult_observed(self):
        with self.assertRaises(ValueError):
            roc_auc(
                np.zeros(4),
                np.ones(4) * 0.1
        )


class Test_logloss(unittest.TestCase):
    def test_logloss(self):
        observed = np.array([1, 1, 1, 0, 0, 0])
        predicted = np.array([0.9, .8, .7, .6, .5, .4])
        self.assertAlmostEqual(
            logloss(observed, predicted), 
            0.4675738
        )

    def test_logloss_trim(self):
        # does the trimming of extreme values work?
        observed = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        predicted1 = np.array([1. - 1e-7, 0.9, .8, .7, .6, .5, .4, 1e-7])
        predicted2 = np.array([1. - 1e-6, 0.9, .8, .7, .6, .5, .4, 1e-6])
        self.assertEqual(
            logloss(observed, predicted1, trim=1e-6), 
            logloss(observed, predicted2, trim=1e-6)
        )

    def test_normLL(self):
        observed = np.array([1, 1, 1, 1, 1, 0])
        predicted = np.array([0.9, .8, .7, .6, .5, .4])
        self.assertAlmostEqual(
            normLL(logloss(observed, predicted), np.mean(observed)),
            0.1122266
        )


class Test_logit(unittest.TestCase):
    def test_logit(self):
        probs = np.array([0.1, 0.5, 0.7])
        log_odds = np.log(np.array([1./9, 1., 7./3]))
        self.assertListEqual(
            rounded_list(logit(probs)),
            rounded_list(log_odds)
        )

    def test_ilogit(self):
        probs = np.array([0.1, 0.5, 0.7])
        log_odds = np.log(np.array([1./9, 1., 7./3]))
        self.assertListEqual(
            rounded_list(probs),
            rounded_list(ilogit(log_odds))
        )


class Test_shuffle_rows(unittest.TestCase):
    def test_rows_shuffled_together(self):
        data = np.arange(100).reshape((100, 1))
        targets = np.arange(100)
        shuf_data, shuf_targets = shuffle_rows(data, targets)
        # if they're shuffled togther all rows should still be equal
        self.assertEqual(
            list(shuf_data.reshape(100)),
            list(shuf_data)
        )

