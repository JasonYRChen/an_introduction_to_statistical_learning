import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe
from decision_tree import DecisionTreeClassifier


np.set_printoptions(precision=3)


class _BaseForest:
    def __init__(self, decision_tree_class, decision_tree_para, 
                 tree_number=100, in_bag_ratio=0.632,
                 is_classifier=True, prediction_agg_func='max_freq'):
        self._class = decision_tree_class # class of decision tree to use
        self._para = decision_tree_para # dict to save paras for decision tree
        self._tree_no = tree_number # number of trees to build
        self._in_bag_ratio = in_bag_ratio # ratio of samples to pick each tree
        self._forest = None # going to be array of decision trees
        self._dummy_oob = None # out-of-bag dummy matrix; 1 for oob, 0 for in bag
        self._multiprocess = False # to use multiprocess. True for yes
        self._is_clsf = is_classifier # True for classifier, False for regressor
        self._agg_func = prediction_agg_func # 'max_freq' or 'mean' as default
            # _agg_func is a prediction aggregate function used to aggregate
            # predictions from each tree in self._forest. It can be a string
            # to specify two default functions: 'max_freq' refers to maximum
            # appearance in predictions, which is normally used in classifiers,
            # and 'mean' refers to the mean of predictions, which is used in
            # regressors. It can also be a user-defined function that takes two
            # parameters: 'y_hats' contains predictions from the forest, and
            # 'dummies' indicates which are the valid predictions in 'y_hats'.
            # The user-defined function then aggregate 'y_hats' along axis = 1 to 
            # get the aggregated prediction.

    @staticmethod
    def _fit_in_multiprocess(conn, _class, _class_paras, X, y, is_categorical):
        # a proxy to build decision tree through multiprocess
        tree = _class(**_class_paras)
        tree.fit(X, y, is_categorical)
        conn.send(tree)
        conn.close()

    @staticmethod
    def _predict_in_multiprocess(conn, tree, X):
        # a proxy to predict X through multiprocess each tree
        y_hat = tree.predict(X)
        conn.send(y_hat)
        conn.close()

    def fit(self, X, y, is_categorical=None, do_oob_test=True):
        rows, cols = X.shape

        # initialize out-of-bag indicator
        self._dummy_oob = np.zeros((rows, self._tree_no), dtype=int)

        # to serve as sampling in observations or features; returning indices
        indices_sample = np.arange(rows)
        indices_feature = np.arange(cols)

        if is_categorical is None: # means every feature is non-categorical
            is_categorical = np.zeros(cols)

        in_bag_no = round(self._in_bag_ratio * rows) # sample number in each tree
       
        if self._multiprocess: # run multiprocess
            conns = [Pipe() for _ in range(self._tree_no)] # pipe to pass tree
            processes = [None for _ in range(self._tree_no)] # to save process
            for i in range(self._tree_no): # every process runs a decision tree
                # draw samples with replacement
                sample_idx = np.random.choice(indices_sample, in_bag_no, True)

                # save out-of-bag indices
                oob = np.delete(indices_sample, sample_idx)
                self._dummy_oob.append(oob)

                # start multiprocessing
                processes[i] = mp.Process(target=self._fit_in_multiprocess,
                     args=(conns[i][1], self._class, self._para, X[sample_idx],
                           y[sample_idx], is_categorical))
                processes[i].start()

            # wait for children processes to finish
            for i in range(self._tree_no):
                processes[i].join()

            # retrieve decision trees from children processes via pipes
            self._forest = [conns[i][0].recv() for i in range(self._tree_no)]

            # close parent-side connection
            for i in range(self._tree_no):
                conns[i][0].close()

        else: # single process
            self._forest = [self._class(**self._para)\
                            for _ in range(self._tree_no)] # initialize forest
            for i in range(self._tree_no):
                # draw samples with replacement
                sample_idx = np.random.choice(indices_sample, in_bag_no, True)

                # indicate in-bag indices. Transform to oob after the loop
                self._dummy_oob[sample_idx, i] = 1 # these are in-bag, not oob

                # save fitting results
                self._forest[i].fit(X[sample_idx], y[sample_idx], is_categorical)

        self._dummy_oob = 1 - self._dummy_oob # transform to oob

        if do_oob_test: # out-of-bag error calculation
            oob_error = self._oob_test(X, y)
            return oob_error 

        return None # normally this method return nothing

    def _aggregate_predictions(self, y_hats, dummies=None):
        # aggregate predictions of all trees
        if dummies is None:
            dummies = np.ones_like(y_hats)

        if self._agg_func == 'max_freq':
            y_hat = np.empty(y_hats.shape[0])
            for i in range(len(y_hat)):
                # dummies[i]==1 could be empty, which means no predictions by
                # the forest on corresponding X[i]. This can crash np.argmax.
                v, c = np.unique(y_hats[i, dummies[i]==1], return_counts=True)
                y_hat[i] = v[np.argmax(c)]
        elif self._agg_func == 'mean':
            for i in range(len(y_hat)):
                y_hat[i] = y_hats[i, dummies[i]==1].mean(1)
        else:
            y_hat = self._agg_func(y_hats, dummies)

        return y_hat

    def prediction_error(self, y_hat, y):
        if self._is_clsf:
            error = np.sum(y_hat != y) / len(y)
        else:
            error = (np.sum((y_hat - y)**2))**0.5 / len(y)
    
        return error

    def predict(self, X):
        if self._multiprocess: # multiprocess on
            conns = [Pipe() for _ in range(self._tree_no)] # pipe to pass y_hat
            processes = [None for _ in range(self._tree_no)] # to save process
            for i in range(self._tree_no):
                # start multiprocessing
                processes[i] = mp.Process(target=self._predict_in_multiprocess,
                    args=(conns[i][1], self._forest[i], X))
                processes[i].start()

            # wait for children processes to finish
            for i in range(self._tree_no):
                processes[i].join()

            y_hats = np.array([conns[i][0].recv() for i in range(self._tree_no)]).T

        else: # single process
            y_hats = np.empty((X.shape[0], len(self._forest)))
            for i in range(self._tree_no):
                y_hats[:, i] = self._forest[i].predict(X)

        y_hat = self._aggregate_predictions(y_hats) # aggregate y_hat

        return y_hat

    def _oob_test(self, X, y):
        # do out-of-bag error. Suitable for multiprocess
        dummies = self._dummy_oob
        y_hats = np.empty_like(dummies) # to save predictions from all trees

        for i, tree in enumerate(self._forest):
            dummy = dummies[:, i]
            y_hats[dummy==1, i] = tree.predict(X[dummy==1])

        y_hat = self._aggregate_predictions(y_hats, dummies)
        error = self.prediction_error(y_hat, y)

        return error


class BaggingClassifier(_BaseForest):
    def __init__(self, tree_number=100, in_bag_ratio=0.632, 
                 impurity_function='gini'):
        dtc_para = {'function': impurity_function}
        super().__init__(DecisionTreeClassifier, dtc_para, 
                         tree_number, in_bag_ratio)


class RandomForestClassifier(_BaseForest):
    def __init__(self, tree_number=100, in_bag_ratio=0.632, 
                 random_feature_ratio=None, impurity_function='gini'):
        dtc_para = {'random_feature_ratio': random_feature_ratio,
                    'function': impurity_function}
        super().__init__(DecisionTreeClassifier, dtc_para, 
                         tree_number, in_bag_ratio)


if __name__ == '__main__':
    trees = 1000
    in_bag_ratio = 0.632
    feat_ratio = None
    function = 'gini'
    multiprocess_on = False

    bg = BaggingClassifier(trees, in_bag_ratio, function)
    rf = RandomForestClassifier(trees, in_bag_ratio, feat_ratio, function)

#    X = np.zeros((25, 3))
    X = np.random.rand(25, 10)
    X[:, 0] = np.random.normal(-1, 1, size=25)
    X[:, 1] = np.random.normal(1, 1, size=25)
    X[:, 2] = np.random.normal(2, 0.5, size=25)
    X[:, 3] = np.random.randint(2, size=25)
    y = np.zeros(25)
    cond1 = X[:, 0] > -1 
    cond2 = (X[:, 1] > 1) & (X[:, 2] < 2)
    y[cond1] = 1
    y[cond2] = 2
    is_categorical = np.zeros(10)
    is_categorical[3] = 1
#    is_categorical = None

    # Out-of-bag error during fitting
    oob_bg = bg.fit(X, y, is_categorical)
    oob_rf = rf.fit(X, y, is_categorical)

    # prediction with X
    y_bg = bg.predict(X)
    y_rf = rf.predict(X)

    # error with X
    e_bg = bg.prediction_error(y_bg, y)
    e_rf = rf.prediction_error(y_rf, y)

    # print errors
    print(f'OOB(bagging): {oob_bg*100:.2f}%')
    print(f'OOB(randfrs): {oob_rf*100:.2f}%')
    print(f'training error(bagging): {e_bg*100:.2f}%')
    print(f'training error(randfrs): {e_rf*100:.2f}%')