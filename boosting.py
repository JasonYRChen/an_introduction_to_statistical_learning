import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from decision_tree import DecisionTreeRegressor
from random_forest_and_bagging import RandomForestRegressor
from random_forest_and_bagging import RandomForestClassifier
from pprint import pprint


class BoostingRegressor:
    def __init__(self, tree_number=1000, learning_rate=0.01, pruning_rate=0,
                 max_depth=1, random_feature_ratio=1, in_bag_ratio=1,
                 max_leaves=None):

        self._tree_number = tree_number
        self._in_bag_ratio = in_bag_ratio # like in bagging
        self._learning_rate = learning_rate # (0, inf]
        self._trees = None # to be container of decision trees

        self._class = DecisionTreeRegressor
        self._tree_para = {'pruning_rate': pruning_rate,
                           'max_depth': max_depth,
                           'random_feature_ratio': random_feature_ratio,
                           'max_leaves': max_leaves}

    def fit(self, X, y, is_categorical=None, sample_weights=None):
        rows, cols = X.shape

        residual = y.copy() # residual of unfitted portion. y could be 2D
        if len(residual.shape) == 1:
            residual = residual.reshape(-1, 1)

        # if multiclass, multiple trees needed in single fitting loop
        trees_per_loop = 1 if len(y.shape) == 1 else y.shape[1]

        # prepare for randomly picking samples in each tree's fitting
        in_bag_no = round(self._in_bag_ratio * rows) # no. of samples used in tree
        indices_sample = np.arange(rows) # full indices to sample 

        self._trees = [[] for _ in range(trees_per_loop)] # container for trees

        for i in range(self._tree_number):
            for j in range(trees_per_loop):
                tree = self._class(**self._tree_para) # initiate new tree
                self._trees[j].append(tree) # save it

                # sample without replacement
                indices = np.random.choice(indices_sample, in_bag_no, False)
                if sample_weights is not None:
                    sample_weights = sample_weights[indices]

                tree.fit(X[indices], residual[:, j][indices], is_categorical, 
                         sample_weights)

                # predict and renew residual
                y_hat_j = tree.predict(X[indices])
                residual[:, j][indices] -= self._learning_rate * y_hat_j

    def predict(self, X):
        y_hat = np.zeros((X.shape[0], len(self._trees)))
        for j, trees in enumerate(self._trees):
            for tree in trees:
                y_hat[:, j] += self._learning_rate * tree.predict(X)

        return y_hat


class BoostingClassifier(BoostingRegressor):
    def fit(self, X, y, is_categorical=None, sample_weights=None):
        """
            A dummy matrix is formed to specify the class of y. For example, if
            y = 2 then on the same row the third column will record as 1, the
            rests on the same row remain 0. The 1 and 0 can also imply odds of
            the observation being a specific class. 1 for 100% of class y, and
            0 for 0% of class y.
        """

        y_unique = np.unique(y) # unique class of y
        y_class = np.zeros((len(y), len(y_unique))) # dummy matrix specify class
        for i in y_unique:
            y_class[:, i][y == i] = 1

        super().fit(X, y_class, is_categorical, sample_weights)

    def predict(self, X):
        """
            Return the class(column) with maximum probability
        """

        y_hat = super().predict(X).argmax(1)
        return y_hat


if __name__ == '__main__':
    samples = 200 
    features = 6
    n_info = 3
    n_class = 3

    tree = 100
    in_bag_boost = 1
    feat_ratio_boost = 1
    depth = np.inf
    learn = 0.01
    prune = 0.05

    in_bag_rf = 0.632
    feat_ratio_rf = None

    # regression
#    X, y = make_regression(samples, features, n_informative=n_info)
#    X_train, X_test, y_train, y_test = train_test_split(X, y)

#    br = BoostingRegressor(tree*10, learn, prune, depth, feat_ratio_boost, in_bag_boost)
#    rf = RandomForestRegressor(tree, in_bag_rf, feat_ratio_rf)

    # fitting
#    br.fit(X_train, y_train)
#    rf.fit(X_train, y_train)

    # predict
#    y_br = br.predict(X_test).flatten()
#    y_rf = rf.predict(X_test)

    # error
#    error = lambda y, y_hat: np.sum((y - y_hat) ** 2)
#    e_br = error(y_test, y_br)
#    e_rf = error(y_test, y_rf)

#    print(f'mse (boost):  {e_br:.0f}')
#    print(f'mse (random): {e_rf:.0f}')

    # classification
    X, y = make_classification(samples, features, n_informative=n_info, n_classes=n_class)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    bc = BoostingClassifier(tree*10, learn, prune, depth, feat_ratio_boost, in_bag_boost)
    rf = RandomForestClassifier(tree, in_bag_rf, feat_ratio_rf)

    # fitting
    bc.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # predict
    y_bc = bc.predict(X_test)
    y_rf = rf.predict(X_test)

    # error
    error = lambda y, y_hat: np.sum(y != y_hat) / len(y)
    e_bc = error(y_test, y_bc)
    e_rf = error(y_test, y_rf)

    print(f'error (boost):  {e_bc*100:.2f}%')
    print(f'error (random): {e_bc*100:.2f}%')
