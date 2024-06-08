import numpy as np
import matplotlib.pyplot as plt


def k_nearest_neighbors(k, x, X, y):
    """
        k: int, number of nearest neighbors to 'x'.
        x: list like, 1 x m of test observation.
        X: ndarray, n x m of known observations.
        y: ndarray, n x 1 of classes' labels of each observation in 'X'.

        returns:
          nearests: ndarray, the nearest neighbors. May be more than one neighbor.
          n: int, number of final number of nearest neighbors, which may be larger
                  than original k.
          odds: float, the odd of test observation to be in the group of 'nearest'
    """

    n = k
    X = np.array(X)
    X -= x
    X **= 2
    distances = X.sum(1)
    sorted_arg = np.argsort(distances) # if k is comparably small to len(y),
                                       # then np.argsort is not efficient.
    while n < len(y) and distances[sorted_arg[n-1]] == distances[sorted_arg[n]]:
        n += 1 # n may be larger than assigned if there are multiple kth smallest
    neighbors = y[sorted_arg[:n]] # filter out the n neighbors

    # conditional probabilities
    y_unique, counts = np.unique(neighbors, return_counts=True)

    # find the equal nearest neighbors
    i = 1
    sorted_arg = np.argsort(counts)
    while i < len(counts) and counts[sorted_arg[-i]] == counts[sorted_arg[-i-1]]:
        i += 1 

    nearests = y_unique[sorted_arg[-i:]]
    odds = counts[sorted_arg[-1]] / sum(counts)
    return nearests, n, odds


x = np.array([1, 2, 3])
X = np.random.randint(2, 7, 60).reshape(-1, 3)
#y = np.random.randint(1, 4, X.shape[0])
y = np.array(['a', 'b', 'c', 'd'] * 15)

np.random.shuffle(y)
print(X)
print(y)
print(k_nearest_neighbors(5, x, X, y))
