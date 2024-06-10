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


if __name__ == '__main__':
#    x = np.array([1, 2, 3])
#    X = np.random.randint(2, 7, 60).reshape(-1, 3)
#    #y = np.random.randint(1, 4, X.shape[0])
#    y = np.array(['a', 'b', 'c', 'd'] * 15)

#    np.random.shuffle(y)
#    print(X)
#    print(y)
#    print(k_nearest_neighbors(5, x, X, y))

    # graphic demo
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    data = 180
    mesh = 60
    x = [0.7, 0]
    k = 5

    # dataset
    x = np.array(x)
    X = np.random.rand(data, 2)
    X[:, 0] = X[:, 0] * (x_max - x_min) + x_min
    X[:, 1] = X[:, 1] * (y_max - y_min) + y_min
    y = np.where(X[:, 0]**2/4+X[:, 1]**2/2.25 < 1, 1, -1)
    y[(X[:, 0] > -0.5) & (X[:, 0] < 0.5)] = -1

    # make meshgrid
    xx, yy = np.linspace(x_min, x_max, mesh), np.linspace(y_min, y_max, mesh)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.flatten()
    yy = yy.flatten()

    # fit and predict meshgrid
    y_grid = [k_nearest_neighbors(k, (i, j), X, y)[0][0] for i, j in zip(xx, yy)]
    y_grid = np.array(y_grid)
    print('x fitting result:')
    print(k_nearest_neighbors(k, x, X, y))

    # scatter plot
    y_color = np.where(y > 0, 'b', 'r')
    y_grid_color = np.where(y_grid > 0, 'b', 'r')
    plt.scatter(X[:, 0], X[:, 1], c=y_color, s=5) # training data
    plt.scatter(xx, yy, c=y_grid_color, s=0.1) # meshgrid
    plt.scatter(x[0], x[1], c='g', s=30) # x

    plt.show()
