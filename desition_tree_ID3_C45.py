from utils import information_gain, information_gain_ratio, argmax, entropy
from collections import Counter
from rich.console import Console
from rich.table import Table


# DT in id3 or c4.5 with no pruning.
class ID3_C45:
    class Node:
        def __init__(self, col, Y):
            self.col = col
            self.children = {}
            self.cnt = Counter(Y)
            self.label = self.cnt.most_common(1)[0][0]

    def __init__(self, information_gain_threshold=0, verbose=False):
        self.information_gain_threshold = information_gain_threshold
        self.verbose = verbose

    def fit(self, X, Y):
        self.column_cnt = len(X[0])
        self.root = self.build(X, Y, set())

    def build(self, X, Y, selected):
        cur = self.Node(None, Y)
        if self.verbose:
            print("cur selected columns: ", selected)
            print("current data: ")
            print(X)
            print(Y)
        split = False
        if len(selected) != self.column_cnt and len(set(Y)) > 1:
            left_columns = list(set(range(self.column_cnt)) - selected)
            # col_idx, best_info_gain = argmax(left_columns, key=lambda col: information_gain(X, Y, col))  # ID3
            col_idx, best_info_gain = argmax(left_columns, key=lambda col: information_gain_ratio(X, Y, col))  # C4.5
            col = left_columns[col_idx]
            if best_info_gain > self.information_gain_threshold:
                split = True
                cur.col = col
                for val in set(x[col] for x in X):
                    idx = [x[col] == val for x in X]
                    child_X = [x for i, x in zip(idx, X) if i]
                    child_Y = [y for i, y in zip(idx, Y) if i]
                    cur.children[val] = self.build(child_X, child_Y, selected | {col})
        if not split and self.verbose:
            print("No split")
        return cur

    def query(self, root, x):
        if root.col is None or x[root.col] not in root.children:
            return root.label
        return self.query(root.children[x[root.col]], x)

    def _predict(self, x):
        return self.query(self.root, x)

    def predict(self, X):
        return [self._predict(x) for x in X]


# Prune DT
def prune(root, X, Y, alpha=.0, verbose=False):
    """
    prune a decision tree recursively. alpha is the weight of tree size in the loss function
    return the loss of all the leaf nodes
    """
    # calculate the entropy of this subtree if the children of root is trimmed
    pruned_entropy = len(X) * entropy(Counter(Y).values())
    pruned_loss = pruned_entropy + alpha
    # if root is a leaf node, return loss directly
    if not root.children:
        return pruned_loss
    cur_loss = 0
    # trim child nodes recursively
    for col_val in root.children:
        child = root.children[col_val]
        idx = [x[root.col] == col_val for x in X]
        childX = [x for i, x in zip(idx, X) if i]
        childY = [y for i, y in zip(idx, Y) if i]
        cur_loss += prune(child, childX, childY, alpha, verbose)
    if pruned_loss < cur_loss:
        root.children.clear()
        return pruned_loss
    return cur_loss


if __name__ == "__main__":
    console = Console(markup=False)
    # id3 = ID3_C45(verbose=False)
    c45 = ID3_C45(verbose=False)
    # -------------------------- Example 1 ----------------------------------------
    # un_pruned decision tree predict correctly for all training data
    print("Example 1:")
    X = [
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
    ]
    Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']

    c45.fit(X, Y)
    pred = c45.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

    # -------------------------- Example 2 ----------------------------------------
    # but un_pruned decision tree doesn't generalize well for test data
    print("Example 2:")
    X = [
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '否', '否', '一般'],
    ]
    Y = ['否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否']
    c45.fit(X, Y)

    testX = [
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
    ]
    testY = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']

    # show in table
    pred = c45.predict(testX)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(testX, testY, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)









