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

import numpy as np
from util import logloss, ilogit

class Loss(object):
    @staticmethod
    def loss(observed, predicted):
        raise NotImplementedError

    @staticmethod
    def gradient(data, pred, targets):
        raise NotImplementedError

class Logistic(Loss):
    name = "LogLoss"

    @staticmethod
    def loss(observed, predicted):
        return logloss(observed, ilogit(predicted), trim=1e-9)

    @staticmethod
    def gradient(data, pred, targets):
        # returns the _positive_ gradient, make it negative when adjusting coefs
        predictions_diff = ilogit(pred) - targets
        coefs_gradient = predictions_diff.dot(data) / len(targets)
        bias_gradient = np.mean(predictions_diff)
        return coefs_gradient, bias_gradient


# For classification trees where we want to compute logloss
# without transformation from the log-odds scale
class Logistic_no_transform(Logistic):
    name = "LogLoss"

    @staticmethod
    def loss(observed, predicted):
        return logloss(observed, predicted, trim=1e-9)


class L2(Loss):
    name = "L2_Loss"

    @staticmethod
    def loss(observed, predicted):
        return np.sqrt(0.5 * np.mean((observed - predicted) ** 2))

    @staticmethod
    def gradient(data, pred, targets):
        # looks very similar to logloss except we don't do ilogit(pred)
        predictions_diff = pred - targets
        coefs_gradient = predictions_diff.dot(data) / len(targets)
        bias_gradient = np.mean(predictions_diff)
        return coefs_gradient, bias_gradient
