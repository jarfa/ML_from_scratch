from __future__ import absolute_import, print_function, division, unicode_literals
import sys
import os
from subprocess import Popen, PIPE
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

from mlfromscratch.util import (
                normalize,
                logloss,
                roc_auc,
                normLL,
)
from mlfromscratch.RegressionSGD import RegressionSGD
from mlfromscratch.RegressionTree import RegressionTree
from mlfromscratch.RandomForest import RandomForest

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def create_models():
    "Return a list of untrained models"
    sgd_args = dict(learning_rate=0.02, minibatch=1, l2=0.0001,
                    n_epochs=100, verbose=False, holdout_proportion=0.0,
                    normalize_data=False)
    tree_args = dict(max_depth=6, min_samples_split=20, min_samples_leaf=10)
    n_forest_trees = 50
    forest_args = dict(max_depth=4, max_features=12, min_samples_split=20,
                        min_samples_leaf=10)
    # Note on the SGD args:
    # Minibatch sizes >1 work better, but sklearn's SGD solver doesn't
    # make it easy to use minibatches. I'm using minibatch=1 just to
    # make the comparison easy.
    # Sklearn also has an algo called LogisticRegression, but SGDClassifier
    # is more directly comparable to mine.
    return [
        (
            ("Logistic Regression", "from scratch"),
            RegressionSGD(loss="logloss", **sgd_args)
        ),
        (
            ("Logistic Regression", "sklearn"),
            SGDClassifier(loss="log", penalty="l2", alpha=sgd_args["l2"],
                learning_rate="constant", eta0=sgd_args["learning_rate"],
                n_iter=sgd_args["n_epochs"])
        ),
        (
            ("Decision Tree", "from scratch"),
            RegressionTree(loss="logloss", **tree_args)
        ),
        (
            ("Decision Tree", "sklearn"),
            DecisionTreeClassifier(criterion="entropy", **tree_args)
        ),
        (
            ("Random Forest", "from scratch"), 
            RandomForest(loss="logloss", num_trees=n_forest_trees,
                **forest_args)
        ),
        (
            ("Random Forest", "sklearn"),
            RandomForestClassifier(criterion="entropy", n_estimators=n_forest_trees,
                **forest_args)
        ),
        # TODO: add gradient boosting when it's done
    ]


def try_model(model, train_data, test_data, train_targets, test_targets):
    "Train a model, test it on holdout data, return metrics"
    start_train = time()
    try:
        # sklearn calls the method 'fit', I'm stubborn and call it train
        model.train(train_data, train_targets)
    except AttributeError:
        model.fit(train_data, train_targets)
    end_train = time()
    test_pred = model.predict_proba(test_data)
    # sklearn's output often has 2 columns, but since this is binary
    # prediction we only need 1.
    if len(test_pred.shape) == 2:
        test_pred = test_pred[:,1]
    end_pred = time()

    test_ll = logloss(test_targets, test_pred)
    test_roc = roc_auc(test_targets, test_pred)

    # for fun, let's look at training error also
    train_pred = model.predict_proba(train_data)
    if len(train_pred.shape) == 2:
        train_pred = train_pred[:,1]
    train_ll = logloss(train_targets, train_pred)
    train_roc = roc_auc(train_targets, train_pred)   

    train_time = end_train - start_train
    pred_time = end_pred - end_train
    return test_ll, test_roc, train_ll, train_roc, train_time, pred_time  


def test_models(csv_name):
    metrics_cols = ["model", "source", "target", "roc_auc", "norm_ll",
        "train_roc_auc", "train_norm_ll", "train_time", "pred_time"]
    metrics_data = dict((k,[]) for k in metrics_cols)

    digits = datasets.load_digits() #has attributes digits.data, digits.target

    # for each target, run each model 3 times on different datasets
    for run in range(3):
        for target_val in range(10):
            # to see how these models compete on a wider variety of data,
            # let's get a different train/test split for each run
            np.random.seed(10 * run + target_val)
            (train_data, holdout_data, train_targets, holdout_targets
                ) = train_test_split(
                    digits.data, 
                    np.array(digits.target == target_val, dtype=float),
                    test_size=0.25
            )
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            norm_train_data = normalize(train_data, train_mean, train_std)
            norm_holdout_data = normalize(holdout_data, train_mean, train_std)
            test_br = np.mean(holdout_targets)
            train_br = np.mean(train_targets)
            # create all models fresh, ready to be trained
            for (mod_name, source), mod in create_models():
                ll, roc, train_ll, train_roc, ttime, ptime = try_model(
                    mod,
                    norm_train_data,
                    norm_holdout_data,
                    train_targets,
                    holdout_targets
                )
                metrics_data["model"].append(mod_name)
                metrics_data["source"].append(source)
                metrics_data["target"].append(target_val)
                metrics_data["roc_auc"].append(roc)
                metrics_data["norm_ll"].append(normLL(ll, test_br))
                metrics_data["train_roc_auc"].append(train_roc)
                metrics_data["train_norm_ll"].append(normLL(train_ll, train_br))
                metrics_data["train_time"].append(ttime)
                metrics_data["pred_time"].append(ptime)

    df = pd.DataFrame(metrics_data)
    df.to_csv(csv_name, index=False)
    print("Wrote {0:d} rows to {1}".format(df.shape[1], csv_name))

if __name__ == "__main__":
    csv_name = os.path.abspath(sys.argv[1])
    if os.path.exists(csv_name):
        print("{0} exists, will skip re-creating that file".format(csv_name))
    else:
        print("Benchmarking...")
        test_models(csv_name)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    r_path = os.path.join(dir_path, "compare_models.R")
    plots_dir = os.path.join(dir_path, "plots/")
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    r_cmd = ["Rscript", "--vanilla", r_path, csv_name, plots_dir]
    print(" ".join(r_cmd))
    status = Popen(r_cmd, stderr=PIPE).wait()
    if status != 0:
        raise Exception("Status: {0}".format(status))

    print("Wrote plots to {}".format(plots_dir))
