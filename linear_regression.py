import numpy as np
import matplotlib.pyplot as plt
#from linear_algebra.singular_value_decomposition import singular_value_decomposition as svd


def linear_regression(matrix, y, add_intercept=True):
    """
        Using singular value decomposition to deal with linear-dependent columns
        in matrix.
    """

    if add_intercept:
        ones = np.ones(matrix.shape[0]).reshape(matrix.shape[0], 1)
        matrix = np.hstack((ones, matrix))

#    u, s, v = svd(matrix)
    u, d, v = np.linalg.svd(matrix)
    d = 1 / d
    s = np.zeros((u.shape[1], v.shape[0]))
    for i, value in enumerate(d):
        s[i, i] = value
    s = s.T
    coefficients = v @ s @ u.T @ y
    return coefficients

if __name__ == '__main__':
    x = np.arange(10).reshape(-1, 1)
    y = 2 * x + 3 + np.random.rand(10).reshape(-1, 1) * 5
    c = linear_regression(x, y)
    print(f'c0, c1: {c}')

    plt.scatter(x, y)
    plt.plot(x, c[0] + c[1] * x, 'r')
    plt.show()
