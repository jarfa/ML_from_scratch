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
import numpy as np
from sklearn.model_selection import train_test_split
from mlfromscratch.util import logloss, normLL, roc_auc, report
from mlfromscratch.RegressionTree import RegressionTree


class RandomForest():
    def __init__(
        self,
        num_trees,
        loss="logloss",
        min_samples_leaf=None,
        min_samples_split=None,
        max_depth=None,
        max_features=None):

        self.num_trees = num_trees
        # format args for subsidiary trees nicely, drop those that aren't specified
        # so that the default RegressionTree args can be used
        all_args = [
            ("loss", loss),
            ("min_samples_leaf", min_samples_leaf),
            ("min_samples_split", min_samples_split),
            ("max_depth", max_depth),
            ("max_features", max_features),
        ]
        self.tree_args = dict((k,v) for k,v in all_args if v is not None)
        self.trees = []

    def _check_trained(self):
        if not self.trees:
            raise AttributeError("Not trained yet.")

    def all_predictions(self, data):
        self._check_trained()
        return np.vstack([t.predict(data) for t in self.trees])

    def predict(self, data):
        pred = self.all_predictions(data)
        r, c = pred.shape
        if r == 1:
            return pred.reshape(c)
        return pred.mean(axis=0)

    def predict_proba(self, data):
        # This implementation doesn't need a separate predict_proba
        # method, but I'm adding it to make let my demonstration
        # code have an easier time looping through models
        return self.predict(data)

    def _train_one(self, data, targets):
        bootstrap_indices = np.random.randint(len(targets), size=len(targets))
        tree = RegressionTree(**self.tree_args)
        tree.train(data[bootstrap_indices,:], targets[bootstrap_indices])
        return tree

    def train(self, data, targets):
        # TODO: look into whether it's possible to skip the GIL and parallelize this
        self.trees = [self._train_one(data, targets) for _ in range(self.num_trees)]


if __name__ == "__main__":
    import argparse
    from time import time
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=int, help='which number to target',
                        required=True)
    parser.add_argument('--loss', choices=['logloss', 'l2'], default='logloss')
    parser.add_argument('--holdout', type=float, default=0.2, 
                        help='holdout proportion (0, 1.0)')
    parser.add_argument('-n', '--num_trees', type=int, required=True)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=None)
    parser.add_argument('--min_samples_leaf', type=int, default=None)
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
    start_forest_train = time()
    forest = RandomForest(
        loss=args.loss,
        num_trees=args.num_trees,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
        max_features=args.max_features)
    
    forest.train(train_data, train_targets)
    end_forest_train = time()

    forest_pred = forest.predict(holdout_data)
    end_forest_pred = time()
    holdout_ll = logloss(holdout_targets, forest_pred)
    report("from scratch",
        end_forest_train - start_forest_train,
        end_forest_pred - end_forest_train,
        normLL=normLL(holdout_ll, np.mean(holdout_targets)),
        roc_auc=roc_auc(holdout_targets, forest_pred)
    )

    # Compare to sklearn's implementation
    start_skl_train = time()
    skl_forest = RandomForestClassifier(
        n_estimators=args.num_trees,
        criterion="entropy",
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split or 20,
        min_samples_leaf=args.min_samples_leaf or 10,
        max_features=args.max_features,
        )
    skl_forest.fit(train_data, train_targets)
    end_skl_train = time()
    
    skl_pred = skl_forest.predict_proba(holdout_data)[:,1]
    end_skl_pred = time()
    skl_ll = logloss(holdout_targets, skl_pred)
    report("scikit-learn",
        end_skl_train - start_skl_train,
        end_skl_pred - end_skl_train,
        normLL=normLL(skl_ll, np.mean(holdout_targets)),
        roc_auc=roc_auc(holdout_targets, skl_pred)
    )
