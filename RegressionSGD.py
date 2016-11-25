from __future__ import absolute_import, print_function, division, unicode_literals
import json
import numpy as np
from sklearn.model_selection import train_test_split
from util import normalize, logloss, normLL, ilogit, shuffle_rows

def report(epoch, ll, br, bias):
    print(
        "Epoch: {epoch:<3} holdout_loss: {mean_loss:.3f} normalized: {norm_loss:.3f} bias: {bias:.3f}".format(
        epoch=epoch,
        mean_loss=ll,
        norm_loss=normLL(ll, br),
        bias=bias
    ))

class LogisticSGD():
    def __init__(self, learning_rate, minibatch=1, coef_init="zero", l1=0, l2=0):
        if coef_init != "zero":
            raise NotImplementedError("Weights can only be initialized at zero for now.")
        self.learning_rate = learning_rate
        self.minibatch = minibatch
        self.coef_init = coef_init
        self.coefs = None
        self.bias = 0.0
        self.l1 = l1
        self.l2 = l2

    def predict(self, data):
        return self.bias + data.dot(self.coefs)

    def predict_prob(self, data):
        return ilogit(self.predict(data))

    def logloss_gradient(self, data, targets):
        # returns the _positive_ gradient, make it negative when adjusting coefs
        predictions_diff = self.predict_prob(data) - targets
        coefs_gradient = predictions_diff.dot(data) / len(targets)
        bias_gradient = np.mean(predictions_diff)
        return coefs_gradient, bias_gradient

    def get_minibatches(self, data, targets):
        for start in range(0, len(targets), self.minibatch):
            yield (data[start:(start + self.minibatch), :],
                targets[start:(start + self.minibatch)])
        
    def train(self, data, targets, n_epochs=1, holdout_proportion=0.2, normalize_data=False):
        # now that we have the data, we know the shape of the weight vector
        self.coefs = np.zeros(data.shape[1])
        # generate holdout set
        train_data, holdout_data, train_targets, holdout_targets = train_test_split(
            data, targets, test_size=holdout_proportion)

        if normalize_data:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_data = normalize(train_data, train_mean, train_std)
            holdout_data = normalize(holdout_data, train_mean, train_std)

        for epoch in range(n_epochs):
            if epoch > 0:
                # randomize order for each epoch
                train_data, train_targets = shuffle_rows(train_data, train_targets)

            for batch_data, batch_targets in self.get_minibatches(train_data, train_targets):
                # evalute the gradient on this minibatch with the current coefs
                w_gradient, b_gradient = self.logloss_gradient(batch_data, batch_targets)
                self.coefs -= self.learning_rate * w_gradient
                self.bias -= self.learning_rate * b_gradient
                # TODO: add learning rate decay
                # TODO: add momentum, rmsprop, etc.

                # regularization
                # should I be regularizing the bias?
                if self.l2:
                    self.coefs -= 2. * self.l2 * self.coefs
                    self.bias -= 2. * self.l2 * self.bias
                if self.l1:
                    self.coefs = np.sign(self.coefs) * np.maximum(
                        0.0, np.absolute(self.coefs) - self.l1)
                    self.bias = np.sign(self.bias) * np.maximum(
                        0.0, np.absolute(self.bias) - self.l1)

            # report after every 2^(n-1) epoch and at the end of training
            if (epoch & (epoch - 1)) == 0 or epoch == (n_epochs - 1):
                # evaluate holdout set w/ current coefs
                holdout_ll = logloss(holdout_targets, self.predict_prob(holdout_data))
                report(
                    epoch=1 + epoch,
                    ll=holdout_ll,
                    br=np.mean(holdout_targets),
                    bias=self.bias
                )


if __name__ == "__main__":
    import argparse
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='step size')
    parser.add_argument('-m', '--minibatch', type=int, default=1, help='minibatch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='# epochs')
    parser.add_argument('-t', '--target', type=int, help='which number to target')
    parser.add_argument('--holdout', type=float, default=0.2, 
                        help='holdout proportion (0, 1.0)')
    parser.add_argument('--l1', type=float, default=0.0, help="L1 Regularization")
    parser.add_argument('--l2', type=float, default=0.0, help="L2 Regularization")
    parser.add_argument('-s', '--seed', type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # We're using the digits data, choosing 1 digit as the target
    # and the rest as non-targets.
    digits = datasets.load_digits()
  
    print("=" * 50)
    print("My model, args: {}".format(json.dumps(args.__dict__, sort_keys=True)))
    print("-" * 50)
    model = LogisticSGD(
        learning_rate=args.learning_rate,
        minibatch=args.minibatch,
        l1=args.l1,
        l2=args.l2,
    )
    model.train(
        digits.data,
        np.array(digits.target==args.target, dtype=float),
        n_epochs=args.epochs,
        holdout_proportion=args.holdout,
        normalize_data=True,
    )

    # Compare to sklearn's equivalent models
    print("=" * 50)
    print("sklearn.linear_model.LogisticRegression")
    print("-" * 50)
    (train_data, holdout_data, train_targets, holdout_targets
        ) = train_test_split(
            digits.data, 
            np.array(digits.target==args.target, dtype=float),
            test_size=args.holdout
    )

    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    train_data = normalize(train_data, train_mean, train_std)
    holdout_data = normalize(holdout_data, train_mean, train_std)

    # I'm using default parameters for sklearn, perhaps this is an unfair comparison?
    sklearn_logistic = LogisticRegression().fit(train_data, train_targets)
    report(
        epoch=sklearn_logistic.n_iter_[0],
        ll=logloss(holdout_targets,
            sklearn_logistic.predict_proba(holdout_data)[:,1]),
        br=np.mean(holdout_targets),
        bias=sklearn_logistic.intercept_[0]
    )
