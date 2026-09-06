import numpy as np
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Table


class naiveBayesMLE:
    def __init__(self, verbose=False):
        self.pa_y = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.py = defaultdict(lambda: 0)
        self.verbose = verbose

    def fit(self, X, Y):
        y_cnt = Counter(Y)
        for x, y in zip(X, Y):
            for i, a in enumerate(x):
                self.pa_y[y][i][a] += 1 / y_cnt[y]
            self.py[y] += 1 / len(X)
        if self.verbose:
            for y in self.py:
                print(f'The prior probability of label {y} is {self.py[y]}.')
                for nth in self.pa_y[y]:
                    prob = self.pa_y[y][nth]
                    for a in prob:
                        print(f'When the label is {y}, the probability that {nth}th attribute be {a} is {prob[a]}.')

    def _predict(self, x):
        labels = list(self.pa_y.keys())
        probs = []
        for y in labels:
            prob = self.py[y]
            for i, a in enumerate(x):
                prob *= self.pa_y[y][i][a]
            probs.append(prob)
        if self.verbose:
            for y, p in zip(labels, probs):
                print(f'The likelihood {x} belongs to {y} is {p}')
        return labels[np.argmax(probs)]

    def predict(self, X):
        return [self._predict(x) for x in X]


class naiveBayesMAP:
    def __init__(self, lamda=1, verbose=False):
        self.pa_y = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.)))
        self.py = defaultdict(lambda: 0.)
        self.lamda = lamda
        self.verbose = verbose

    def fit(self, X, Y):
        y_cnt = Counter(Y)
        for col in range(len(X[0])):
            col_values = set(x[col] for x in X)
            for x, y in zip(X, Y):
                self.pa_y[y][col][x[col]] += 1
            for y in y_cnt:
                for a in self.pa_y[y][col]:
                    self.pa_y[y][col][a] += self.lamda
                    self.pa_y[y][col][a] /= y_cnt[y] + self.lamda * len(col_values)
        for y in y_cnt:
            self.py[y] = (y_cnt[y] + self.lamda) / (len(X) + self.lamda * len(y_cnt))
        if self.verbose:
            for y in self.pa_y:
                print(f'The prior probability of label {y} is', self.py[y])
                for nth in self.pa_y[y]:
                    prob = self.pa_y[y][nth]
                    for a in prob:
                        print(f'When the label is {y}, the probability that {nth}th attribute be {a} is {prob[a]}')

    def _predict(self, x):
        labels = list(self.pa_y.keys())
        probs = []
        for y in labels:
            prob = self.py[y]
            for i, a in enumerate(x):
                prob *= self.pa_y[y][i][a]
            probs.append(prob)
        if self.verbose:
            for y, p in zip(labels, probs):
                print(f'The likelihood {x} belongs to {y} is {p}')
        return labels[np.argmax(probs)]

    def predict(self, X):
        return [self._predict(x) for x in X]


if __name__ == "__main__":
    console = Console(markup=False)
    naive_bayes_mle = naiveBayesMLE(verbose=True)
    naive_bayes_map = naiveBayesMAP(verbose=True)

# -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = [
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
    ]
    Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    naive_bayes_mle.fit(X, Y)
    naive_bayes_map.fit(X, Y)

    pred_mle = naive_bayes_mle.predict(X)
    pred_map = naive_bayes_map.predict(X)
    table = Table('x', 'y', 'pred_mle', 'pred_map')
    for x, y, y_hat_mle, y_hat_map in zip(X, Y, pred_mle, pred_map):
        table.add_row(*map(str, [x, y, y_hat_mle, y_hat_map]))
    console.print(table)






