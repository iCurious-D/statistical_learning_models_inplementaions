import numpy as np
from matplotlib import pyplot as plt
from utils import Heap, euc_dis

class KDtree:
    class Node:
        def __init__(self, points, labels, axis):
            self.points = points
            self.labels = labels
            self.axis = axis
            self.left = None
            self.right = None

    def __init__(self, X, Y):
        self.root = self.build(X, Y)

    def build(self, X, Y, split_axis=0):
        if not len(X):
            return None
        median_ind = np.argpartition(X[:, split_axis], len(X)//2, axis=0)[len(X)//2]
        split_point = float(X[median_ind, split_axis])
        equal_x = X[X[:, split_axis] == split_point]
        equal_y = Y[X[:, split_axis] == split_point]
        less_x = X[X[:, split_axis] < split_point]
        less_y = Y[X[:, split_axis] < split_point]
        greater_x = X[X[:, split_axis] > split_point]
        greater_y = Y[X[:, split_axis] > split_point]
        node = self.Node(equal_x, equal_y, split_axis)
        node.left = self.build(less_x, less_y, 1-split_axis)
        node.right = self.build(greater_x, greater_y, 1-split_axis)
        return node

    def _query(self, root, x, k):
        if not root:
            return Heap(max_len=k, key=lambda xy: -euc_dis(x, xy[0]))
        if x[root.axis] <= root.points[0][root.axis]:
            ans = self._query(root.left, x, k)
            sibling = root.right
        else:
            ans = self._query(root.right, x, k)
            sibling = root.left
        for curx, cury in zip(root.points, root.labels):
            ans.push((curx, cury))
        if len(ans) < k or -ans.top_key() > abs(x[root.axis] - root.points[0][root.axis]):
            other_ans = self._query(sibling, x, k)
            while other_ans:
                otherx, othery = other_ans.pop()
                ans.push((otherx, othery))
        return ans

    def query(self, x, k):
        return self._query(self.root, x, k)


class KNN:
    def __init__(self, k, distance_func="l2"):
        self.k = k
        if distance_func == "l2":
            self.distance_func = lambda x, y: np.linalg.norm(x-y)
        else:
            self.distance_func = distance_func

    def fit(self, X, Y):
        self.tree = KDtree(X, Y)
        self.k = min(self.k, len(X))

    def _predict(self, x):
        top_k = self.tree.query(x, self.k)
        top_k_y = [y for x, y in top_k]
        return np.argmax(np.bincount(top_k_y))

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)


if __name__ == "__main__":
    def demonstrate(X_train, Y_train, X_test, k, desc):
        knn = KNN(k=k)
        knn.fit(X_train, Y_train)
        pred_test = knn.predict(X_test)

        # plot
        plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_test, marker=".", s=1)
        plt.title(desc)
        plt.show()

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 2, 3, 4, 5])
    # generate grid-shaped test data
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
