import warnings
from copy import deepcopy
from operator import add
import numpy as np

def combine_dicts(A, B, fn=add):
    # http://stackoverflow.com/a/11011911/3011972
    return {x: fn(A.get(x, 0), B.get(x, 0)) for x in set(A).union(B)}

def logloss_gradient(target, prediction):
    return np.mean(prediction - target)

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

        self.weights = {"bias": 0.0}

    def predict(self, event):
        return sum(self.weights.get(k, 0.0) * v for k,v in event.iteritems())

    def predict_prob(self, event):
        return ilogit(self.predict(event))

    def get_minibatches(self, data):
        for start in range(0, len(data), self.minibatch):
            # copy to avoid messing up future epochs
            yield deepcopy(data[start:(start + self.minibatch)])
        
    def train(self, data, n_epochs=1, holdout_every=10):
        # add bias 'feature'
        for event in data:
            event["bias"] = 1.0

        # generate holdout set
        np.random.shuffle(data) #make sure there's no regular pattern in the data
        holdout_set = [event for i,event in enumerate(data) if i % holdout_every == 0]
        data = [event for i,event in enumerate(data) if i % holdout_every != 0]

        for epoch in range(n_epochs):
            if epoch > 0:
                np.random.shuffle(data)
            for batch in self.get_minibatches(data):
                n_batch = len(batch) # note: n_batch != self.minibatch on the last iteration
                # evalute this minibatch with the current weights
                targets = np.array([event.pop("target") for event in batch])
                predictions = np.array([self.predict_prob(event) for event in batch])
                gradient = logloss_gradient(targets, predictions) #am I taking the sum of gradients?
                
                # add up value of all features to complete the gradient
                sum_event = reduce(combine_dicts, batch)
                
                # update weights with the latest gradient
                for feat_name, feat_value in sum_event.iteritems():
                    self.weights[feat_name] = (
                        self.weights.get(feat_name, 0.) - 
                        (self.alpha * gradient * feat_value / n_batch)
                    )
    
            # report after every 2^(n-1) epoch and at the end of all epochs
            if (epoch & (epoch - 1)) == 0 or epoch == (n_epochs - 1):
                # evaluate holdout set w/ current weights
                targets = np.array([event["target"] for event in holdout_set])
                predictions = np.array([self.predict_prob(event) for event in holdout_set])
                ll = logloss(targets, predictions)
                print(
                    "Epoch: {epoch}, holdout_loss: {mean_loss:.3f}, normalized: {norm_mean_loss:.3f}, bias: {bias:.3f}".format(
                    epoch=1 + epoch,
                    mean_loss=ll,
                    norm_mean_loss=normLL(ll, np.mean(targets)),
                    bias=self.weights["bias"]
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
    args = parser.parse_args()

    digits = datasets.load_digits()
    paired_data = zip(digits.data, digits.target)

    def row2dict(data_row, outcome, target_val):
        data = dict((i, val) for i,val in enumerate(data_row))
        data["target"] = float(outcome == target_val)
        return data
    data = [row2dict(features, out, args.target) for features, out in paired_data]

    model = LogisticSGD(alpha=args.alpha, minibatch=args.minibatch)
    model.train(data, n_epochs=args.epochs)

    