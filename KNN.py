import matplotlib.pyplot as plt
import numpy as np
from functools import partial


class KNN:
    def __init__(self, k=1, distance_func="l2"):
        self.k = k
        if distance_func == "l2":
            self.distance_func = lambda x, y: np.linalg.norm(x-y)
        else:
            self.distance_func = distance_func

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.k = min(self.k, len(self.X))

    def _knn(self, x):
        dis = np.apply_along_axis(partial(self.distance_func, y=x), axis=-1, arr=self.X)
        top_k_id = np.argpartition(dis, self.k)[:self.k]
        return top_k_id

    def _predict(self, x):
        top_k_id = self._knn(x)
        top_y = self.Y[top_k_id]
        return np.argmax(np.bincount(top_y))

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)


if __name__ == "__main__":
    def demonstrate(X_train, y_train, X_test, k, desc):
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        pred_y = knn.predict(X_test)

        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_y, marker=".", s=1)
        plt.title(desc)
        plt.show()


    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 2, 3, 4, 5])
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    demonstrate(X_train, Y_train, X_test, 1, "Example 1")

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 1, 2, 3, 4])
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    demonstrate(X_train, Y_train, X_test, 1, "Example 2")

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 1, 2, 2, 2])
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    demonstrate(X_train, Y_train, X_test, 1, "Example 3")







