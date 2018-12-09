# Copyright 2017 Jonathan Arfa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division, unicode_literals
import json
from time import time
import numpy as np
from collections import namedtuple
from sklearn.model_selection import train_test_split
from mlfromscratch.util import logloss, normLL, roc_auc, report
from mlfromscratch.loss import Logistic_no_transform, L2

tree_config = namedtuple("TreeConfig",
    ["loss", "min_samples_leaf", "min_samples_split", "max_depth", "max_features"])


def find_potential_splits(data, p=0.05):
    splits = np.percentile(data, 100 * np.arange(p, 1.0, p), axis=0)
    return dict(
        (c, np.unique(splits[:, c])) for c in range(splits.shape[1])
    )


class SubTree():
    def __init__(self, config, potential_splits=None):
        self.config = config  # of type tree_config
        self.children = []
        self.split_feature = None
        self.split_value = None
        self.mean_predict = None
        self.subtree_loss = None
        self.potential_splits = potential_splits

    def is_terminal(self):
        return len(self.children) == 0

    def find_split(self, data, targets):
        if self.potential_splits is None:
            # this should only happen if this is the top node of the tree
            self.potential_splits = find_potential_splits(data, p=0.05)

        # Even when max_features=None, I'm randomizing the order by which splits are
        # considered because otherwise I'd have to find a principled way to break ties
        # between 2 splits that lead to equal loss. This change only marginally slows
        # down the code.
        max_features = (
            len(self.potential_splits) if self.config.max_features is None else
            min(self.config.max_features, len(self.potential_splits))
        )
        feature_indices = np.random.choice(len(self.potential_splits),
                                           max_features, replace=False)
        # TODO: when multiple separations have equal loss, it would be nice to
        # choose the one in the middle. Do other implementations do this?
        for i in feature_indices:
            for s in np.random.permutation(self.potential_splits[i]):
                split_ix = data[:,i] >= s
                if sum(split_ix) in (0, len(split_ix)):
                    # i.e. if splitting the data here is not a real split
                    continue
                predictions = np.zeros_like(targets)
                predictions[split_ix] = np.mean(targets[split_ix])
                predictions[-split_ix] = np.mean(targets[-split_ix])
                s_loss = self.config.loss.loss(targets, predictions)
                if s_loss < self.subtree_loss:
                    self.subtree_loss = s_loss
                    self.split_feature = i
                    self.split_value = s

    def train(self, data, targets):
        self.mean_predict = np.mean(targets)
        self.subtree_loss = self.config.loss.loss(
            targets, self.mean_predict * np.ones_like(targets))

        if (len(data) < self.config.min_samples_split or
            self.config.max_depth == 0):
            return

        self.find_split(data, targets)

        if self.split_feature is None:
            return # There's no good split, this subtree will return the mean value

        self.children = [None, None]
        split_indices = data[:, self.split_feature] >= self.split_value

        if self.config.max_depth is None:
            next_config = self.config
        else:
            next_config = self.config._replace(
                max_depth=self.config.max_depth - 1)

        if (self.config.min_samples_leaf is None or
            sum(-split_indices) >= self.config.min_samples_leaf):
            self.children[0] = SubTree(config=next_config,
                                       potential_splits=self.potential_splits)
            self.children[0].train(
                data[-split_indices, :], targets[-split_indices]
            )
        if (self.config.min_samples_leaf is None or
            sum(split_indices) >= self.config.min_samples_leaf):
            self.children[1] = SubTree(config=next_config,
                                       potential_splits=self.potential_splits)
            self.children[1].train(
                data[split_indices, :], targets[split_indices]
            )

        return self

    def predict(self, data):
        if self.is_terminal():
            return self.mean_predict

        predictions = np.ones(len(data)) * self.mean_predict
        split_indices = data[:, self.split_feature] >= self.split_value

        if self.children[0] is not None:
            predictions[-split_indices] = self.children[0].predict(
                data[-split_indices, :])

        if self.children[1] is not None:
            predictions[split_indices] = self.children[1].predict(
                data[split_indices, :])

        return predictions


class RegressionTree():
    def __init__(
        self,
        loss="logloss",
        min_samples_leaf=10,
        min_samples_split=20,
        max_depth=None,
        max_features=None,
        ):

        self.tree = None
        loss = {"logloss": Logistic_no_transform(), "l2": L2()}.get(loss)
        if loss is None:
            raise ValueError("Loss argument {0} not recognized.".format(loss))

        self.config = tree_config(
            min_samples_leaf = min_samples_leaf,
            min_samples_split = max(min_samples_split, min_samples_leaf),
            max_depth = max_depth,
            max_features = max_features,
            loss=loss
        )


    def _check_trained(self):
        if not self.tree:
            raise AttributeError("Not trained yet.")

    def train(self, data, targets):
        self.tree = SubTree(self.config).train(data, targets)

    def predict(self, data):
        self._check_trained()
        return self.tree.predict(data)


if __name__ == "__main__":
    import argparse
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=int, help='which number to target',
                        required=True)
    parser.add_argument('--loss', choices=['logloss', 'l2'], default='logloss')
    parser.add_argument('--holdout', type=float, default=0.2,
                        help='holdout proportion (0, 1.0)')
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=20)
    parser.add_argument('--min_samples_leaf', type=int, default=10)
    parser.add_argument('--max_features', type=int, default=None)
    parser.add_argument('-s', '--seed', type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)

    digits = datasets.load_digits()

    (train_data, holdout_data, train_targets, holdout_targets
        ) = train_test_split(
            digits.data,
            np.array(digits.target==args.target, dtype=float),
            test_size=args.holdout
    )

    start_tree_train = time()
    tree = RegressionTree(
        loss=args.loss,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
        max_features=args.max_features)

    tree.train(train_data, train_targets)
    end_tree_train = time()

    tree_pred = tree.predict(holdout_data)
    end_tree_pred = time()
    holdout_ll = logloss(holdout_targets, tree_pred)
    report("Tree (from scratch)",
        end_tree_train - start_tree_train,
        end_tree_pred - end_tree_train,
        normLL=normLL(holdout_ll, np.mean(holdout_targets)),
        roc_auc=roc_auc(holdout_targets, tree_pred)
    )

    # Compare to sklearn's implementation
    start_skl_train = time()
    skl_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split or 20,
        min_samples_leaf=args.min_samples_leaf or 10,
        max_features=args.max_features,
        )
    skl_tree.fit(train_data, train_targets)
    end_skl_train = time()

    skl_pred = skl_tree.predict_proba(holdout_data)[:,1]
    end_skl_pred = time()
    skl_ll = logloss(holdout_targets, skl_pred)
    report("sklearn.tree.DecisionTreeClassifier",
        end_skl_train - start_skl_train,
        end_skl_pred - end_skl_train,
        normLL=normLL(skl_ll, np.mean(holdout_targets)),
        roc_auc=roc_auc(holdout_targets, skl_pred)
    )
