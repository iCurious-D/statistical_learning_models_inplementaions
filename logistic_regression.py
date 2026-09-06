import matplotlib.pyplot as plt
import numpy as np
from utils import sigmoid, binary_cross_entropy, wbline


class logisticRegression:
    def __init__(self, max_iteration=1000, lr=1, verbose=True):
        self.lr = lr
        self.verbose = verbose
        self.max_iteration = max_iteration

    def fit(self, X, Y):
        self.feature_size = X.shape[-1]
        self.w = np.random.rand(self.feature_size)
        self.b = np.random.rand(1)
        for iter in range(self.max_iteration):
            pred = self._predict(X)
            # bias gradient of shape [data-size]
            grandient_b = Y - pred
            # weight gradient of shape [data-size, feature]
            grandient_w = grandient_b[:, None] * X
            # get mean of gradient across all data
            gradient_b = grandient_b.mean(axis=0)
            gradient_w = grandient_w.mean(axis=0)
            self.w += gradient_w * self.lr
            self.b += gradient_b * self.lr
            if self.verbose:
                loss = binary_cross_entropy(pred, Y)
                print(f"step {iter}, loss is {loss}...")

    def _predict(self, X):
        logit = self.w @ X.transpose() + self.b
        pred = sigmoid(logit)
        return pred

    def predict(self, X):
        pred = self._predict(X)
        Y = (pred > .5).astype(int)
        return Y


if __name__ == "__main__":
    def demonstrate(X, Y, desc):
        LR = logisticRegression(verbose=True)
        LR.fit(X, Y)

        plt.title(desc)
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        wbline(LR.w, LR.b)
        plt.show()


    # -------------------------- Example 1 ----------------------------------------
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, 0, 0])
    demonstrate(X, Y, "Example 1")

    # -------------------------- Example 2 ----------------------------------------
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 0, 0, 1])
    demonstrate(X, Y, "Example 2: Logistic Regression still cannot solve a simple XOR problem")

    # -------------------------- Example 3 ----------------------------------------
    X1 = np.random.normal([0, 1], size=[40, 2])
    X2 = np.random.normal([1, 0], size=[40, 2])
    X = np.concatenate((X1, X2))
    Y = np.concatenate([np.ones(40), np.zeros(40)])
    demonstrate(X, Y, "Example 3: Logistic Regression is suitable for tasks that are not strictly linear separable")






