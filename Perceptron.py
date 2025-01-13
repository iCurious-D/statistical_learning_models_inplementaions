import numpy as np
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from utils import *

# 感知机学习算法的原始形式
class Perceptron_origin(object):
    def __init__(self, lr=1e-1, max_iteration=2000):
        self.lr = lr
        self.max_iteration = max_iteration

    def _trans(self, x):
        return self.w @ x + self.b

    def _predict(self, x):
        return 1 if self._trans(x) >= 0. else -1

    def fit(self, X, Y):
        self.feature_size = X.shape[-1]
        """ 选取初值 np.random.rand() """
        self.w = np.random.rand(self.feature_size)
        self.b = np.random.rand(1)
        updated = 1
        epoch = 0
        """ 迭代优化 """
        while updated > 0 and epoch < self.max_iteration:
            print(f"epoch {epoch} start; ")
            updated = 0
            # shuffle data
            perm = np.random.permutation(len(X))
            for i in perm:
                x, y = X[i], Y[i]
                """ 错误点驱动参数更新 """
                if self._predict(x) is not y:
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    updated += 1
            print(f"finished at iters: {epoch}, w: {self.w}, b: {self.b}")
            epoch += 1
        """ 已达最大迭代次数情况 """
        print(f"finished for reaching the max_iter: {self.max_iteration}, w: {self.w}, b: {self.b}")
        return

    def predict(self, X):
        return np.apply_along_axis(func1d=self._predict, axis=-1, arr=X)


# 感知机学习算法的对偶形式
class Perceptron_dual(object):
    def __init__(self, lr=1e-1, max_iteration=2000):
        self.lr = lr
        self.max_iteration = max_iteration

    def _cal_w(self, X, y):
        w = 0
        for i in range(len(self.alpha)):
            w += self.alpha[i] * y[i] * X[i]
        return w

    def _trans(self, x):
        return self.w @ x + self.b

    def _predict(self, x):
        return 1 if self._trans(x) >= 0. else -1

    def _gram_matrix(self, X):
        """ 计算xixj内积的 gram 矩阵"""
        return np.dot(X, X.T)

    def fit(self, X, y):
        N, M = X.shape
        """ α 和 β 初始化为0 """
        self.alpha = np.zeros(N)
        self.b = 0
        gram = self._gram_matrix(X)
        epoch = 0
        while epoch < self.max_iteration:
            print(f"epoch {epoch} started...")
            wrong_items = 0
            for i in range(N):
                tmp = 0
                for j in range(N):
                    tmp += self.alpha[j] * y[j] * gram[i, j]
                tmp += self.b
                """ 错误点驱动参数更新 """
                if y[i] * tmp <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
                    wrong_items += 1
            """ 到达终止循环条件：没有误分类点，用 α 计算 w """
            if wrong_items == 0:
                self.w = self._cal_w(X, y)
                print(f"finished at iters: {epoch}, w: {self.w}, b: {self.b}")
                return
            epoch += 1
        """ 已达最大迭代次数情况 """
        self.w = self._cal_w(X, y)
        print(f"finished for reaching the max_iter: {self.max_iteration}, w: {self.w}, b: {self.b}")
        return

    def predict(self, X):
        return np.apply_along_axis(func1d=self._predict, axis=-1, arr=X)


# 测试函数
# 辅助绘图函数
def draw_lines(w, b, *args, **kwargs):
    if w[1] == 0:
        plt.vlines(-b/w[0], *plt.gca().get_ylim(), *args, **kwargs)
    else:
        x_vals = np.array(plt.gca().get_xlim())
        y_vals = (-w[0] / w[1]) * x_vals + b / w[1]
        plt.plot(x_vals, y_vals, *args, **kwargs)

def test_model(X, Y, desc):
    # perceptron = Perceptron_origin()
    perceptron = Perceptron_dual()
    perceptron.fit(X, Y)

    # mathplot
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    wbline(perceptron.w, perceptron.b)
    plt.title(desc)
    plt.show()

    # console print
    console = Console(markup=False)
    pred = perceptron.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)


if __name__ == "__main__":
    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    test_model(X, Y, "Example 1")

    # -------------------------- Example 2 ----------------------------------------
    print("Example 2: Perceptron cannot solve a simple XOR problem")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, -1, -1, 1])
    test_model(X, Y, "Example 2: Perceptron cannot solve a simple XOR problem")


