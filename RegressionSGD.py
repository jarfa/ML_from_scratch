import warnings
from operator import add
import numpy as np

def logloss(observed, predicted):
    # keep loss from being infinite
    predicted = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    return -np.mean(
        observed * np.log(predicted) + 
        (1. - observed) * np.log(1. - predicted)
    )

def normLL(raw_logloss, baserate):
    ll_br = -(baserate * np.log(baserate) + (1 - baserate) * np.log(1 - baserate))
    return 1. - (raw_logloss / ll_br)

def ilogit(log_odds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1. / (1. + np.exp(-log_odds))

def shuffle_rows(data, targets):
    if not data.shape[0] == len(targets):
        raise Exception("Data and targets do not have the same number of rows.")
    shuffle_ix = np.random.permutation(len(targets))
    return data[shuffle_ix,:], targets[shuffle_ix]

class LogisticSGD():
    def __init__(self, alpha, minibatch=1, weight_init="zero", l1=0, l2=0):
        if l1 != 0 or l2 != 0:
            raise NotImplementedError("Regularization ain't here yet.")
        if weight_init != "zero":
            raise NotImplementedError("Weights can only be initialized at zero for now.")
        self.alpha = alpha
        self.minibatch = minibatch
        self.weight_init = weight_init
        self.l1 = l1
        self.l2 = l2

    def predict(self, events):
        return events.dot(self.weights)

    def predict_prob(self, events):
        return ilogit(self.predict(events))

    def logloss_gradient(self, data, targets):
        predictions = self.predict_prob(data)
        return (predictions - targets).dot(data) / self.minibatch

    def get_minibatches(self, data, targets):
        N = len(targets)
        for start in range(0, N, self.minibatch):
            yield (data[start:(start + self.minibatch),], 
                targets[start:(start + self.minibatch)])
        
    def train(self, targets, data, n_epochs=1, holdout_proportion=0.2):
        Ntotal, Nfeat = data.shape
        self.weights = np.zeros(Nfeat + 1)
        # add bias column
        data = np.hstack((np.ones((Ntotal,1)), data))
        
        # generate holdout set
        Nholdout = round(holdout_proportion * Ntotal)
        data, targets = shuffle_rows(data, targets)
        
        holdout_data = data[:Nholdout,:]
        holdout_targets = targets[:Nholdout]
        train_data = data[Nholdout:,:]
        train_targets = targets[Nholdout:]

        for epoch in range(n_epochs):
            if epoch > 0:
                train_data, train_targets = shuffle_rows(train_data, train_targets)
            for batch_data, batch_targets in self.get_minibatches(train_data, train_targets):
                # evalute this minibatch with the current weights
                gradient = self.logloss_gradient(batch_data, batch_targets)
                self.weights -= self.alpha * gradient

                # TODO: L1 & L2
    
            # report after every 2^(n-1) epoch and at the end of training
            if (epoch & (epoch - 1)) == 0 or epoch == (n_epochs - 1):
                # evaluate holdout set w/ current weights
                ll = logloss(holdout_targets, self.predict_prob(holdout_data))
                print(
                    "Epoch: {epoch:<3} holdout_loss: {mean_loss:.3f} normalized: {norm_mean_loss:.3f} bias: {bias:.3f}".format(
                    epoch=1 + epoch,
                    mean_loss=ll,
                    norm_mean_loss=normLL(ll, np.mean(holdout_targets)),
                    bias=self.weights[0]
                    ))



if __name__ == "__main__":
    import argparse
    from sklearn import datasets
    np.random.seed(5)

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.01, help='step size')
    parser.add_argument('-m', '--minibatch', type=int, default=1, help='minibatch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='# epochs')
    parser.add_argument('-t', '--target', type=int, help='which number to target')
    parser.add_argument('--holdout', type=float, 
        help='holdout proportion (0, 1.0)', default=0.2)
    args = parser.parse_args()

    digits = datasets.load_digits()
  
    model = LogisticSGD(alpha=args.alpha, minibatch=args.minibatch)
    model.train(np.array(digits.target==args.target, dtype=float), digits.data,
        n_epochs=args.epochs, holdout_proportion=args.holdout)
    