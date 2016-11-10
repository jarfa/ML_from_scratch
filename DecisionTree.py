from __future__ import absolute_import, print_function, division, unicode_literals
import json
import numpy as np
from sklearn.model_selection import train_test_split
from util import logloss, normLL

def find_potential_splits(data, p=0.05):
    splits = np.percentile(data, np.arange(p * 100, 100, p * 100), axis=0)
    return dict((c, splits[:, c]) for c in range(splits.shape[1]))

def find_split(data, targets, potential_splits, leaf_only=False, max_features=None):
    best_split = {
        "N": len(targets),
        "mean_predict": np.mean(targets),
        "subtree_loss": np.inf, 
        "split_feature": None,
        "split_value": None, 
        "children": [None, None],
    }
    if leaf_only:
        best_split["subtree_loss"] = logloss(
            targets,
            best_split["mean_predict"] * np.ones_like(targets)
        )
        return best_split

    if max_features is None:
        feature_indices = range(len(potential_splits))
    else:
        feature_indices = np.random.choice(len(potential_splits), max_features,
            replace=False)

    for i in feature_indices:
        for s in potential_splits[i]:
            split_ix = (data[:,i] >= s)
            if sum(split_ix) in (0, len(split_ix)):
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

class DecisionTree():
    # I'm borrowing some of the variable names from sklearn's implementation
    def __init__(self, min_samples_split=5, loss="logloss", max_depth=None,
        max_features=None):
        # self.min_samples_leaf = min_samples_leaf # TODO
        self.min_samples_split = min_samples_split
        if loss != "logloss":
            raise NotImplementedError("Need to make more loss functions")
        self.loss = loss
        self.potential_splits = None
        self.tree = None
        self.max_depth = max_depth
        self.max_features = max_features

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

    def build_subtree(self, data, targets, depth):
        if (self.max_depth is not None and depth >= self.max_depth
            ) or data.shape[0] <= self.min_samples_split:
            return find_split(data, targets, None, leaf_only=True)

        split = find_split(data, targets, self.potential_splits, max_features=self.max_features)
        split_indices = data[:,split["split_feature"]] >= split["split_value"]
        split["children"][0] = self.build_subtree(data[-split_indices, :], targets[-split_indices], depth=depth+1)
        split["children"][1] = self.build_subtree(data[split_indices, :], targets[split_indices], depth=depth+1)
        return split

    def train(self, data, targets):
        self.potential_splits = find_potential_splits(data, p=0.05)
        self.tree = self.build_subtree(data, targets, depth=0)

    def args_dict(self):
        return {
            "max_depth": self.max_depth,
            "max_features": self.max_features,
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
    import argparse
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    np.random.seed(5)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=int, help='which number to target')
    parser.add_argument('--holdout', type=float, default=0.2, 
                        help='holdout proportion (0, 1.0)')
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=20)
    parser.add_argument("--max_features", type=int, default=None)
    args = parser.parse_args()

    digits = datasets.load_digits()
    
    (train_data, holdout_data, train_targets, holdout_targets
        ) = train_test_split(
            digits.data, 
            np.array(digits.target==args.target, dtype=float),
            test_size=args.holdout
    )

    tree = DecisionTree(
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
