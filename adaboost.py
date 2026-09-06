import numpy as np
from math import log
from rich.console import Console
from rich.table import Table


class DecisionStump:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def fit(self, X, Y, weight):
        X = X[:, 0]
        possible_thresholds = list(set(X))
        possible_thresholds.append(max(possible_thresholds) + 1)
        possible_thresholds.append(min(possible_thresholds) - 1)
        best_acc = 0
        best_threshold, best_sign = 0.,  0.
        for sign in [1, -1]:
            self.sign = sign
            for threshold in possible_thresholds:
                self.threshold = threshold
                pred = self.predict(X)
                acc = (pred == Y) @ weight
                if acc > best_acc:
                    best_acc, best_threshold, best_sign = acc, self.threshold, self.sign
        self.threshold, self.sign = best_threshold, best_sign
        if self.verbose:
            print(f"Threshold is {self.threshold}")

    def predict(self, X):
        X = X * self.sign
        threshold = self.threshold * self.sign
        pred = (X > threshold) * 2 - 1
        return pred.flatten()


class AdaBoost:
    def __init__(self, basic_model=DecisionStump, max_nums=10, verbose=True):
        self.verbose = verbose
        self.BasicModel = basic_model
        self.max_nums = max_nums
        self.basic_models = []
        self.model_weights = []

    def fit(self, X, Y):
        n = len(X)
        data_weight = np.ones(n) / n
        for i in range(self.max_nums):
            basic_model = self.BasicModel()
            basic_model.fit(X, Y, data_weight)
            self.basic_models.append(basic_model)
            pred = basic_model.predict(X)
            error_rate = (pred != Y) @ data_weight
            model_weight = .5 * log((1-error_rate)/error_rate)
            self.model_weights.append(model_weight)
            data_weight *= np.exp(-model_weight * Y * pred)
            data_weight /= data_weight.sum()
            if self.verbose:
                print(f"number {i}, current error rate is {error_rate}")
                print(f"The weight of current model is {model_weight}")

    def predict(self, X):
        score = sum(model.predict(X) * model_weight for model, model_weight in zip(self.basic_models, self.model_weights))
        pred = (score > 0.).astype(int) * 2 - 1
        return pred


if __name__ == "__main__":
    def demonstrate(X, Y, desc):
        print(desc)
        console = Console(markup=False)
        adaboost = AdaBoost(verbose=True)
        adaboost.fit(X, Y)

        pred = adaboost.predict(X)
        table = Table('X', 'Y', 'pred')
        for x, y, y_hat in zip(X, Y, pred):
            table.add_row(*map(str, [x, y, y_hat]))
        console.print(table)

    X = np.arange(10).reshape(-1, 1)
    Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    demonstrate(X, Y, "example 1")
            




