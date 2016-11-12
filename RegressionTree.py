from __future__ import absolute_import, print_function, division, unicode_literals
import json
import numpy as np
from sklearn.model_selection import train_test_split
from util import logloss, normLL

def find_potential_splits(data, p=0.05):
    splits = np.percentile(data, 100 * np.arange(p, 1.0, p), axis=0)
    return dict(
        (c, np.unique(splits[:, c])) for c in range(splits.shape[1])
    )

# I'm borrowing some of the variable names from sklearn's implementation
class RegressionTree():
    def __init__(
        self,
        loss="logloss",
        min_samples_leaf=10,
        min_samples_split=20,
        max_depth=None,
        max_features=None,
        ):
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = max(min_samples_split, min_samples_leaf)
        if loss != "logloss":
            raise NotImplementedError("Need to make more loss functions")
        self.loss = loss
        self.max_depth = max_depth
        self.max_features = max_features
        self.potential_splits = None
        self.tree = None

    def _predict_event(self, event):
        subtree = self.tree
        while(True):
            feat, val = subtree["split_feature"], subtree["split_value"]
            if feat is None:
                return subtree["mean_predict"]

            child = 1 if event[feat] >= val else 0
            if subtree["children"][child] is None:
                return subtree["mean_predict"]
            
            subtree = subtree["children"][child]    

    def predict(self, data):
        if len(data.shape) == 1 or data.shape[1] == 1:
            return self._predict_event(data)
        return np.apply_along_axis(self._predict_event, 1, data)

    def find_split(self, data, targets, leaf_only=False):
        mean_predict = np.mean(targets)
        best_split = {
            "N": len(targets),
            "mean_predict": mean_predict,
            # if no split can beat logloss(baserate_predictor), don't consider it
            "subtree_loss": logloss(targets, mean_predict * np.ones_like(targets)),
            "split_feature": None,
            "split_value": None,
            "children": [None, None],
        }
        if leaf_only:
            return best_split

        # Even when max_features=None, I'm randomizing the order by which splits are
        # considered because otherwise I'd have to find a principled way to break ties
        # between 2 splits that lead to equal loss. This change only marginally slows
        # down the code.
        max_features = self.max_features or len(self.potential_splits)
        feature_indices = np.random.choice(len(self.potential_splits), max_features,
                            replace=False)
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
                ll = logloss(targets, predictions)
                if ll < best_split["subtree_loss"]:
                    best_split["subtree_loss"] = ll
                    best_split["split_feature"] = i
                    best_split["split_value"] = s

        return best_split

    def build_subtree(self, data, targets, depth):
        if (self.max_depth is not None and depth >= self.max_depth
            ) or data.shape[0] <= self.min_samples_split:
            return self.find_split(data, targets, leaf_only=True)

        split = self.find_split(data, targets, leaf_only=False)
        if split["split_feature"] is None:
            return split
        split_indices = data[:,split["split_feature"]] >= split["split_value"]
        if sum(-split_indices) >= self.min_samples_leaf:
            split["children"][0] = self.build_subtree(data[-split_indices, :], targets[-split_indices], depth=depth+1)
        if sum(split_indices) >= self.min_samples_leaf:
            split["children"][1] = self.build_subtree(data[split_indices, :], targets[split_indices], depth=depth+1)
        return split

    def train(self, data, targets):
        self.potential_splits = find_potential_splits(data, p=0.05)
        self.tree = self.build_subtree(data, targets, depth=0)

    def args_dict(self):
        return {
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "loss": self.loss,
        }

    def __repr__(self):
        if not self.tree:
            raise AttributeError("Not trained yet.")
        return json.dumps(
            (self.args_dict(), self.tree),
            indent=4, sort_keys=True)


if __name__ == "__main__":
    import argparse
    from sklearn import datasets

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=int, help='which number to target',
                        required=True)
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

    tree = RegressionTree(
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
        max_features=args.max_features)
    
    tree.train(train_data, train_targets)
    
    holdout_ll = logloss(holdout_targets, tree.predict(holdout_data))
    print(
        "Norm LL: {nll}".format(
            nll=normLL(holdout_ll, np.mean(holdout_targets))
            )
    )
