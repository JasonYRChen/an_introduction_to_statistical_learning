import heapq
import numpy as np
from collections import deque


np.set_printoptions(precision=3)

_ZERO = 10 ** (-10) # any abs(value) smaller than this is regarded as 0


def _X_category_partition(feature, y, impurity_func, sample_weights):
    category = np.unique(feature)
    criterion, impurity = None, np.inf
    for c in category:
        selection = feature == c
        impurity_left = impurity_func(y[selection], sample_weights[selection])
        impurity_right = impurity_func(y[~selection], sample_weights[~selection])

        len_selection, len_feature = len(feature[selection]), len(feature)
        n_left, n_right = len_selection, len_feature - len_selection
        impurity_global = (n_left * impurity_left + n_right * impurity_right) / len_feature
        # truncate to zero if very small
        if abs(impurity_global) <= _ZERO:
            impurity_global = 0

        if impurity_global < impurity:
            criterion, impurity = c, impurity_global

    return impurity, criterion


def _X_number_partition(feature, y, impurity_func, sample_weights):
    sorted_idx = np.argsort(feature)
    criterion, impurity = None, np.inf
    for i in range(len(feature) + 1):
        left_i, right_i = sorted_idx[:i], sorted_idx[i:]
        impurity_left = impurity_func(y[left_i], sample_weights[left_i])
        impurity_right = impurity_func(y[right_i], sample_weights[right_i])

        n_left, n_right = i, len(feature) - i
        impurity_global = (n_left * impurity_left + n_right * impurity_right) / len(feature)
        # truncate to zero if very small
        if abs(impurity_global) <= _ZERO:
            impurity_global = 0

        if impurity_global < impurity:
            impurity = impurity_global
            if i == 0:
                criterion = feature[sorted_idx[0]] - 0.5
            elif i == len(feature):
                criterion = feature[sorted_idx[-1]] + 0.5
            else:
                criterion = (feature[sorted_idx[i-1]] + feature[sorted_idx[i]]) / 2

    return impurity, criterion


def _decision_stump(X, y, impurity_func, is_categorical=None, sample_weights=None):
    # is_categorical: array of {0, 1} or {True, False} to specify corresponding
    #                 feature (column) in X is categorical or not.
    rows, cols = X.shape

    if is_categorical is None:
        is_categorical = np.zeros(cols)

    if sample_weights is None:
        sample_weights = np.ones(rows)
    sample_weights /= sum(sample_weights) # to make sure the sum is 1

    feature_no, criterion, sign = None, None, None
    impurity = np.inf
    for ft in range(cols):
        func = _X_category_partition if is_categorical[ft] else _X_number_partition
        e_temp = '==' if is_categorical[ft] else '<='

        l_temp, s_temp = func(X[:, ft], y, impurity_func, sample_weights)
        if l_temp < impurity:
            feature_no, criterion, impurity, sign = ft, s_temp, l_temp, e_temp

    return feature_no, criterion, sign


class Tree:
    class _Node:
        def __init__(self, indices, depth=0):
            self.indices = indices # indices of elements
            self.depth = depth # depth of node in a tree. Valid depth start from 1
            self.feature = None # column number
            self.sign = '' # '==' or '<='
            self.criterion = None # to partition node into subnodes
            self.impurity = None # impurity of the node
            self.parent = None # parent node
            self.left = None # left child node
            self.right = None # right child node
            self.y_hat = None # to save predicted y

        def is_leaf(self):
            if self.left == None and self.right == None:
                return True
            return False

        def __repr__(self):
            return f'Node(\'feature {self.feature}\' {self.sign} ' +\
                   f'{self.criterion:.3f}, depth={self.depth}, ' +\
                   f'impurity={self.impurity:.3f}, prediction={self.y_hat})' +\
                   f', id={id(self)}, parent={id(self.parent)}, ' +\
                   f'sibling={id(self._sibling())}'

        def __len__(self):
            return len(self.indices)

        def _sibling(self):
            if self.parent:
                condition = self != self.parent.left
                return self.parent.left if condition else self.parent.right
            return None

        def __lt__(self, node):
            # this method is meant for heapq
            # Currently comparing two nodes is meaningless
            if not isinstance(node, self.__class__):
                raise TypeError(f'{node} is not a valid _Node instance')
            return -self.depth < -node.depth

    def __init__(self,max_depth=np.inf):
        self.root = None
        self.max_depth = max_depth # 0 for no limit
        self.leaves = deque()
        self.X = None # if no need, delete it
        self.y = None # if no need, delete it

    def create_node(self, indices=None, depth=0):
        return self._Node(indices, depth=depth)

    def _link_nodes(self, parent, left_child, right_child):
        parent.left, parent.right = left_child, right_child
        left_child.parent = right_child.parent = parent
        return parent

    def __repr__(self):
        return f'Tree(root={self.root}, max depth: {self.max_depth})'

    def expand_tree(self, node=None):
        if node is None:
            node = self.root

        print(f'{" "*2*node.depth}{node}')
        if node.left:
            print(f'{" "*2*node.depth}--if feature {node.feature} {node.sign} {node.criterion}')
            self.expand_tree(node.left)
            inv_sign = '!=' if node.sign == '==' else '>'
            print(f'{" "*2*node.depth}--if feature {node.feature} {inv_sign} {node.criterion}')
            self.expand_tree(node.right)


class DecisionTreeClassifier:
    def __init__(self, pruning_rate=0, function='gini', max_depth=np.inf,
                 num_random_feature=None):
        self.tree = Tree(max_depth)
        self.pruning_rate = pruning_rate
        self._num_random_feature = num_random_feature
        self._impurity_functions = {'entropy': self.loss_entropy, 
                                    'gini': self.loss_gini, 
                                    'classification': self.loss_classification
                                   }
        self._func = self._impurity_functions[function]

    @staticmethod
    def _class_probability(y, sample_weights):
        sample_weights /= sum(sample_weights)
        p = np.array([sum(sample_weights[y==k]) for k in np.unique(y)])
        return p

    def loss_classification(self, y, sample_weights):
        p = self._class_probability(y, sample_weights)
        impurity = 1 - max(p) if p.size else 1
        return impurity 

    def loss_entropy(self, y, sample_weights):
        p = self._class_probability(y, sample_weights)
        p_log = np.log(p) # p may contain 0. Additional measure required
        p_log[p_log == -np.inf] = 0
        impurity = -p @ p_log
        return impurity

    def loss_gini(self, y, sample_weights):
        p = self._class_probability(y, sample_weights)
        impurity = p @ (1 - p)
        return impurity

    @staticmethod
    def _set_prediction(y, nodes):
        for node in nodes:
            indices = node.indices
            values, counts = np.unique(y[indices], return_counts=True)
            node.y_hat = values[np.argmax(counts)]

    @staticmethod
    def _renew_indices(X, indices, feature, criterion, sign):
        # partition indices by sign
        if sign == '==':
            l_idx = np.where(X[indices][:, feature] == criterion)[0]
            r_idx = np.where(X[indices][:, feature] != criterion)[0]
        else:
            assert sign == '<='
            l_idx = np.where(X[indices][:, feature] <= criterion)[0]
            r_idx = np.where(X[indices][:, feature] > criterion)[0]
             
        # indices for left and right sub nodes
        left_indices = indices[l_idx]
        right_indices = indices[r_idx]

        return left_indices, right_indices

    def _total_impurity(self, leaves):
        total_elements = 0
        total_impurity = 0
        for leaf in leaves:
            total_elements += len(leaf.indices)
            total_impurity += len(leaf.indices) * leaf.impurity
        total_impurity /= total_elements

        return total_elements, total_impurity

    def _prune(self):
        # Except for the root, every node must have a sibling. 
        # They must exist or be removed together.

        new_leaves = set(self.tree.leaves) # need to covert to deque at the end
        heap = list(self.tree.leaves)
        heapq.heapify(heap)
        alpha = self.pruning_rate

        total_elements, total_impurity = self._total_impurity(self.tree.leaves)
        penalty = total_impurity + alpha * len(new_leaves)

        while heap:
            node = heapq.heappop(heap)
            if node in new_leaves:
                sibling = node._sibling()

                if sibling is not None and sibling.is_leaf():
                    # only if the node is not the root and its sibling is a
                    # leaf can join the pruning process
                    parent = node.parent
                    new_total_impurity = total_impurity +\
                                      (len(parent.indices)*parent.impurity -\
                                       len(node.indices)*node.impurity -\
                                       len(sibling.indices)*sibling.impurity) /\
                                       total_elements
                    new_penalty = new_total_impurity + alpha*(len(new_leaves)-1)

                    if new_penalty < penalty:
                        penalty, total_impurity = new_penalty, new_total_impurity
                        new_leaves.remove(node)
                        new_leaves.remove(sibling)
                        new_leaves.add(node.parent)
                        heapq.heappush(heap, node.parent)

        new_leaves = deque(new_leaves) # convert to deque to be compatible

        # cut off all the leaves beneath new leaves
        for leaf in new_leaves:
            leaf.left = leaf.right = None

        return new_leaves

    def fit(self, X, y, is_categorical=None, sample_weights=None):
        # 'is_categorical' is an array to specify which features (columns) in
        # 'X' are categorical data

        rows, cols = X.shape

        if is_categorical is None:
            is_categorical = np.zeros(cols)
        
        if sample_weights is None:
            sample_weights = np.ones(rows)

        col_indices = np.arange(cols) # indices of columns
        feature_num = self._num_random_feature

        self.tree.root = self.tree.create_node(np.arange(rows)) # create root
        nodes_to_process = deque([self.tree.root]) # nodes to split

        while nodes_to_process:
            node = nodes_to_process.popleft()
            indices = node.indices

            # to use all the features or random select part of them
            if feature_num is None:
               col_picked = col_indices
            else:
                col_picked = np.random.choice(col_indices, feature_num, False)

            # greedily find partition criterion
            column, criterion, sign = _decision_stump(X[indices][:, col_picked],
                y[indices], self._func, is_categorical, sample_weights[indices])
            feature = col_picked[column]

            # renew node's attributes
            node.feature, node.criterion, node.sign, node.impurity = \
                feature, criterion, sign,\
                self.loss_classification(y[indices], sample_weights[indices])

            # partition indices by sign
            left_indices, right_indices = self._renew_indices(X, indices,
                feature, criterion, sign)

            # nodes to further partition or end as leaves
            if not (left_indices.size and right_indices.size) or\
                node.depth == self.tree.max_depth:
                self.tree.leaves.append(node) # end as leaf
            else:
                # connect parent node and descendants
                self.tree._link_nodes(node,
                    self.tree.create_node(left_indices, node.depth+1),
                    self.tree.create_node(right_indices, node.depth+1))

                # to further partition
                nodes_to_process.append(node.left)
                nodes_to_process.append(node.right)

        # prune the tree if pruning rate > 0
        if self.pruning_rate:
            new_leaves = self._prune()
            self.tree.leaves = new_leaves

        self._set_prediction(y, self.tree.leaves) # prediction of each leaf

    def predict(self, X):
        rows = X.shape[0]
        y_hat = np.empty(rows)
        for i in range(rows):
            node = self.tree.root
            while not node.is_leaf():
                feature, sign, criterion = node.feature, node.sign, node.criterion
                cond1 = (sign == '==') and (X[i, feature] == criterion)
                cond2 = (sign == '<=') and (X[i, feature] <= criterion)
                if cond1 or cond2:
                    node = node.left
                else:
                    assert (X[i, feature] != criterion) or (X[i, feature] > criterion)
                    node = node.right
            y_hat[i] = node.y_hat

        return y_hat


if __name__ == '__main__':
    alpha = 0.1
    row, col = 20, 8
    num_random_feature = 2
    no_of_randint = 3

    dtc_unprune = DecisionTreeClassifier()
    dtc_random = DecisionTreeClassifier(num_random_feature=num_random_feature)
    dtc_prune = DecisionTreeClassifier(pruning_rate=alpha)

    X = np.zeros((row, col))

    # X contains category
#    X[:, 0] = np.random.rand(row)
#    X[:, 1] = np.random.randint(no_of_randint, size=row)
#    X[:, 2] = np.random.normal(size=row)
#    is_categorical = [0, 1, 0]
#    y_cond1 = 'X[:, 1] == 1'
#    y_cond2 = 'X[:, 2] >= 0'

    # X not contain category
    X[:, 0] = np.random.normal(-1, 0.5, row)
    X[:, 1] = np.random.normal(1, 2, row)
    X[:, 2] = np.random.normal(0, 0.5, row)
    is_categorical = np.zeros(col)
    y_cond1 = 'X[:, 0] < -1'
    y_cond2 = 'X[:, 1] > 1'

    # y
    y = np.zeros(row)
    cond1 = eval(y_cond1)
    cond2 = eval(y_cond2)
    y[cond1] = np.trunc(np.random.normal(11, 1, sum(cond1)))
    y[~cond1 & cond2] = np.trunc(np.random.normal(22, 1, len(y[~cond1 & cond2])))

    # sample_weights:
#    sw = np.random.rand(10)
    sw = None

#    print(f'y condition 1: {y_cond1}')
#    print(f'y condition 2: {y_cond2}')
#    print(f'X & y:\n{np.hstack((X, y.reshape(-1, 1)))}')

    print('---------Unpruned----------')
    dtc_unprune.fit(X, y, is_categorical, sw)
    dtc_unprune.tree.expand_tree()

    print('---------random forest like----------')
    dtc_random.fit(X, y, is_categorical, sw)
    dtc_random.tree.expand_tree()

    print('---------Pruned----------')
    dtc_prune.fit(X, y, is_categorical, sw)
    dtc_prune.tree.expand_tree()

    # prediction
#    y_hat = dtc.predict(X)
#    error = np.sum(y_hat != y) / row
#    print(f'\nprediction error with training data: {error*100}%')


## test case for _decision_stump
#def y_classify(y, sample_weights):
#    sample_weights /= sum(sample_weights)
#    p = np.array([sum(sample_weights[y==k]) for k in np.unique(y)])
#    impurity = p @ (1 - p)
#    return impurity
#def y_regress(y, sample_weights):
#    sample_weights /= sum(sample_weights)
#    y_mean = sample_weights @ y
#    mse = sample_weights @ (y - y_mean)**2
#    return mse
#col_classify = 2
#impurity_func = y_classify
##impurity_func = y_regress
#X = (np.random.rand(rows, cols) - 0.5) * 10
#X[:, col_classify] = np.random.randint(3, size=rows)

## y for classification
#y = np.random.randint(1, 3, size=rows)
##y[X[:, 1] > 0] = 0 # for X_number
#y[X[:, 2] == 1] = 0 # for X_classification

## y for regression
##y = np.random.normal(-1, 0.5, size=rows)
##y[X[:, 1] > 0] = np.random.normal(1, 0.5, size=sum(X[:, 1]>0)) # for X_number
##y[X[:, 2] == 1] = np.random.normal(1, 0.5, size=sum(X[:, 2]==1)) # for X_classify

#is_categorical = np.zeros(cols)
#is_categorical[col_classify] = 1
#sw = np.ones(rows)
#print(f'X:\n{X}\ny: {y}\nis category: {is_categorical}\nsample weights: {sw}')
#print(f'data:\n{np.hstack((X, y.reshape(-1, 1)))}\nis category: {is_categorical}\nsample weights: {sw}')
#ft, criterion, sign = _decision_stump(X, y, impurity_func, is_categorical, sw)
#print(f'feature: {ft}, criterion: {criterion}, sign: {sign}')
