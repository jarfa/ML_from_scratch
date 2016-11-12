from __future__ import absolute_import, print_function, division, unicode_literals
import numpy as np
from sklearn.model_selection import train_test_split
from util import logloss, normLL
from RegressionTree import RegressionTree


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
        if loss != "logloss":
            raise NotImplementedError("Need to make more loss functions")
        self.loss = loss
        # format args for subsidiary trees nicely, drop those that aren't specified
        # so that the default RegressionTree args can be used
        self.tree_args = [
            ("min_samples_leaf", min_samples_leaf),
            ("min_samples_split", min_samples_split),
            ("max_depth", max_depth),
            ("max_features", max_features),
        ]
        self.tree_args = dict((k,v) for k,v in self.tree_args if v is not None)
        self.trees = []

    def all_predictions(self, data):
        if not self.trees:
            raise AttributeError("Not trained yet.")
        return np.vstack([t.predict(data) for t in self.trees])

    def predict(self, data):
        return self.all_predictions(data).mean(axis=0)

    # TODO: return more than just the mean prediction - confidence intervals, etc.

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
    from sklearn import datasets
    np.random.seed(5)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=int, help='which number to target',
                        required=True)
    parser.add_argument('--holdout', type=float, default=0.2, 
                        help='holdout proportion (0, 1.0)')
    parser.add_argument('-n', '--num_trees', type=int, required=True)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=None)
    parser.add_argument('--min_samples_leaf', type=int, default=None)
    parser.add_argument('--max_features', type=int, default=None)
    args = parser.parse_args()

    digits = datasets.load_digits()
    
    (train_data, holdout_data, train_targets, holdout_targets
        ) = train_test_split(
            digits.data, 
            np.array(digits.target==args.target, dtype=float),
            test_size=args.holdout
    )

    forest = RandomForest(
        num_trees=args.num_trees,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
        max_features=args.max_features)
    
    forest.train(train_data, train_targets)
    
    holdout_ll = logloss(holdout_targets, forest.predict(holdout_data))
    print(
        "Norm LL: {nll}".format(
            nll=normLL(holdout_ll, np.mean(holdout_targets))
            )
    )
