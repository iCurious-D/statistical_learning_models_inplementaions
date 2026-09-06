from math import nan, inf
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table
from utils import gini


class regressionCART:
    class Node:
        def __init__(self, col, Y):
            self.col = col
            self.val = nan
            self.left, self.right = None, None
            self.label = Y.mean()

        def __hash__(self):
            return id(self)

    def __init__(self, max_depth=inf, verbose=False):
        self.verbose = verbose
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = self.build(X, Y)

    def get_se(self, Y_cnt):
        mean = sum(Y_cnt[y] * y for y in Y_cnt) / sum(Y_cnt.values())
        square_error = sum((y - mean) ** 2 * Y_cnt[y] for y in Y_cnt)
        return square_error

    def get_se_of_split(self, Y1_cnt, Y2_cnt):
        return self.get_se(Y1_cnt) + self.get_se(Y2_cnt)

    def build(self, X, Y, depth=1):
        if self.verbose:
            print("cur data:")
            print(X)
            print(Y)
        cur = self.Node(None, Y)
        best_se = inf
        best_col, best_val = -1, nan
        if depth < self.max_depth and len(set(Y)) > 1:
            for col in range(len(X[0])):
                smaller_Y_cnt = Counter()
                lager_Y_cnt = Counter(Y)
                sorted_idxs = np.argsort(X[:, col])
                for i, idx in enumerate(sorted_idxs):
                    smaller_Y_cnt[Y[idx]] += 1
                    lager_Y_cnt[Y[idx]] -= 1
                    # don't split on the largest number, otherwise the right part is empty
                    if sorted_idxs[i] == sorted_idxs[-1]:
                        break
                    # split only when this is the last one of consequent identical numbers
                    if i == len(X) - 1 or X[idx, col] != X[sorted_idxs[i+1], col]:
                        se = self.get_se_of_split(smaller_Y_cnt, lager_Y_cnt)
                        if se < best_se:
                            val = X[idx, col]
                            best_se, best_col, best_val = se, col, val
            if self.verbose:
                print(f"split by value {best_val} of {best_col}th column")
            smaller_idx = X[:, best_col] <= best_val
            larger_idx = X[:, best_col] > best_val
            smaller_X = X[smaller_idx]
            smaller_Y = Y[smaller_idx]
            larger_X = X[larger_idx]
            larger_Y = Y[larger_idx]
            cur.val = best_val
            cur.col = best_col
            cur.left = self.build(smaller_X, smaller_Y, depth+1)
            cur.right = self.build(larger_X, larger_Y, depth+1)
        elif self.verbose:
            print("No split")
        return cur

    def _query(self, root, x):
        if root.col is None:
            return root
        elif x[root.col] > root.val:
            return self._query(root.right, x)
        return self._query(root.left, x)

    def query(self, root, x):
        return self._query(root, x).label

    def _predict(self, x):
        return self.query(self.root, x)

    def predict(self, X):
        return [self._predict(x) for x in X]

# =====================================================================================


class classificationCART:
    class Node:
        def __init__(self, col, Y):
            self.col = col
            self.val = None
            self.left, self.right = None, None
            self.label = Counter(Y).most_common(1)[0][0]

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, Y):
        self.root = self.build(X, Y)

    def get_gini_of_split(self, Y1, Y2):
        gini_y1 = gini(Y1)
        gini_y2 = gini(Y2)
        length = len(Y1) + len(Y2)
        return len(Y1)/length * gini_y1 + len(Y2) / length * gini_y2

    def build(self, X, Y):
        if self.verbose:
            print("cur data:")
            print(X)
            print(Y)
        cur = self.Node(None, Y)
        best_gini = inf
        best_col, best_val = -1, nan
        if len(set(Y)) > 1:
            for col in range(len(X[0])):
                val_set = set(X[:, col])
                if len(val_set) != 1:
                    for val in val_set:
                        selected_idx = X[:, col] == val
                        other_idx = X[:, col] != val
                        selected_Y = Y[selected_idx]
                        other_Y = Y[other_idx]
                        cur_gini = self.get_gini_of_split(selected_Y, other_Y)
                        if cur_gini < best_gini:
                            best_gini, best_col, best_val = cur_gini, col, val
            selected_idx = X[:, best_col] == best_val
            other_idx = X[:, best_col] != best_val
            selected_X = X[selected_idx]
            selected_Y = Y[selected_idx]
            other_X = X[other_idx]
            other_Y = Y[other_idx]
            cur.col = best_col
            cur.val = best_val
            cur.left = self.build(selected_X, selected_Y)
            cur.right = self.build(other_X, other_Y)
        elif self.verbose:
            print("No split")
        return cur

    def query(self, root, x):
        if root.col is None:
            return root.label
        elif x[root.col] > root.val:
            return self.query(root.right, x)
        return self.query(root.left, x)

    def _predict(self, x):
        return self.query(self.root, x)

    def predict(self, X):
        return [self._predict(x) for x in X]


class prunedCART:
    def __init__(self, cart, X, Y, val_X, val_Y, verbose):
        self.verbose = verbose
        self.root = cart.root
        self.possible_prune_threshold = {np.inf}
        # stage one: calculate pruning loss for all nodes
        self.calculate_prune_loss(self.root, X, Y)
        if self.verbose:
            print("All the possible threshold values are: ", self.possible_prune_threshold)
        # stage two: choose the best threshold for pruning
        self.prune_threshold = self.choose_threshold(val_X, val_Y, self.possible_prune_threshold)
        if self.verbose:
            print("The best threshold value is: ", self.prune_threshold)

    def calculate_prune_loss(self, root, X, Y):
        # calculate the gini index of this subtree if the children of root is trimmed
        pruned_gini = len(X) * gini(Counter(Y).values())
        pruned_loss = pruned_gini
        # if root is a leaf node, return loss directly
        if root.col is None:
            return pruned_loss, 1

        # cur_loss record the loss function when root is not trimmed
        cur_loss = 0
        # size record the size of this subtree
        size = 1

        selected_idx = X[:, root.col] == root.val
        other_idx = X[:, root.col] != root.val
        selected_X = X[selected_idx]
        selected_Y = Y[selected_idx]
        other_X = X[other_idx]
        other_Y = Y[other_idx]

        # trim the left node recursively
        child_loss, child_size = self.calculate_prune_loss(root.left, selected_X, selected_Y)
        cur_loss += child_loss
        size += child_size

        # trim the right node recursively
        child_loss, child_size = self.calculate_prune_loss(root.right, other_X, other_Y)
        cur_loss += child_loss
        size += child_size

        # the loss of prune the branches of this node
        relative_prune_loss = (pruned_loss - cur_loss) / (size - 1)
        root.relative_prune_loss = relative_prune_loss
        self.possible_prune_threshold.add(relative_prune_loss)
        return cur_loss, size

    def choose_threshold(self, val_X, val_Y, possible_prune_threshold):
        """
        Choose the best subtree according to the validation set.
        Cross-validation here simply refers to predict on a pre-split validation set.
        """
        best_acc = -1
        best_prune_threshold = 0
        for prune_threshold in sorted(list(possible_prune_threshold)):
            cur_acc = self.validate(val_X, val_Y, prune_threshold)
            if cur_acc >= best_acc:
                best_acc = cur_acc
                best_prune_threshold = prune_threshold
        return best_prune_threshold

    def validate(self, val_X, val_Y, prune_threshold):
        pred = self.predict(val_X, prune_threshold)
        return (pred == val_Y).mean()

    def predict(self, X, prune_threshold=None):
        if prune_threshold is None:
            prune_threshold = self.prune_threshold
        return np.array([self._predict(x, prune_threshold) for x in X])

    def _predict(self, x, prune_threshold):
        return self.query(self.root, x, prune_threshold)

    def query(self, root, x, prune_threshold):
        if root.col is None or root.relative_prune_loss < prune_threshold:
            return root.label
        elif x[root.col] != root.val:
            return self.query(root.right, x, prune_threshold)
        return self.query(root.left, x, prune_threshold)


# =====================================================================================
if __name__ == "__main__":

# ===== classification CART======================
    console = Console(markup=False)
    cart_clf = classificationCART(verbose=True)
    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = np.array([
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ])
    Y = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])
    cart_clf.fit(X, Y)

    pred = cart_clf.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

    # -------------------------- Example 2 ----------------------------------------
    # but unpruned decision tree doesn't generalize well for test data
    print("Example 2:")
    testX = np.array([
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ])
    testY = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])

    pred = cart_clf.predict(testX)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(testX, testY, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

    # Here I use the same dataset as the validation set
    # Notice that it must be the full tree to be choosed this way
    pruned_cart = prunedCART(cart_clf, X, Y, testX, testY,verbose=False)

    # show in table
    pred = pruned_cart.predict(testX)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(testX, testY, pred):
       table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

# =====regression CART=============================================
    def demonstrate(cart, X, Y, test_X, test_Y, desc):
        print(desc)
        console = Console(markup=False)
        cart.fit(X, Y)
        pred = cart.predict(test_X)
        table = Table('x', 'y', 'pred')
        for x, y, y_hat in zip(test_X, test_Y, pred):
            table.add_row(*map(str,[x, y, y_hat]))
        console.print(table)

    # 习题 5.2=============================================
    cart = regressionCART(verbose=True)
    X = np.arange(1, 11).reshape(-1, 1)
    Y = np.array([4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.90, 8.23, 8.70, 9.00])

    demonstrate(cart, X, Y, X, Y, "example 1:")
    test_X = X + .5
    test_Y = np.zeros_like(Y) + nan
    demonstrate(cart, X, Y, test_X, test_Y, "Example 2:")

    cart1 = regressionCART(verbose=True, max_depth=1)
    demonstrate(cart1, X, Y, X, Y, "example 3: regressionCART stump")

    cart2 = regressionCART(verbose=True, max_depth=3)
    demonstrate(cart2, X, Y, X, Y, "example 4: regressionCART split twice")
