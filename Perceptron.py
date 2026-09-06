import os

from matplotlib import pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from utils import wbline


class Perceptron:
    def __init__(self, lr=1e-1, max_iteration=2000, verbose=False):
        self.lr = lr
        self.verbose = verbose
        self.max_iteration = max_iteration
        self.b = np.random.rand(1)

    def _trans(self, x):
        return self.w @ x + self.b

    def _predict(self, x):
        return 1 if self._trans(x) >= 0 else -1

    def fit(self, X, Y):
        self.feature_size = X.shape[-1]
        self.w = np.random.rand(self.feature_size)
        count = 1
        epoch = 0
        while epoch < self.max_iteration and count > 0:
            if self.verbose:
                print(f"epoch {epoch} started...")
            count = 0
            perm = np.random.permutation(len(X))
            for i in perm:
                x, y = X[i], Y[i]
                if self._predict(x) != y:
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    count += 1
            if self.verbose:
                print(f"epoch {epoch} finished, {count} pieces of data mis-classified")
            epoch += 1
        return

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)


if __name__ == "__main__":
    def demonstrate(X, Y, desc):
        console = Console(markup=False)
        perceptron = Perceptron(verbose=True)
        perceptron.fit(X, Y)

        plt.scatter(X[:, 0], X[:, 1], c=Y)
        wbline(perceptron.w, perceptron.b)
        plt.title(desc)
        plt.show()

        pred = perceptron.predict(X)
        table = Table('x', 'y', 'pred')
        for x, y, y_hat in zip(X, Y, pred):
            table.add_row(*map(str, [x, y, y_hat]))
        console.print(table)

    print("Example 1:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    demonstrate(X, Y, "Example 1")

    print("Example 2: Perceptron cannot solve a simple XOR problem")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, -1, -1, 1])
    demonstrate(X, Y, "Example 2: Perceptron cannot solve a simple XOR problem")
