import numpy as np
import matplotlib.pyplot as plt


class PrincipalComponentRegression:
    def __init__(self, explained_var=0.8):
        self.explained_var = explained_var
        self.coefs = None
        self.eig_vecs = None

    @staticmethod
    def standardize(X):
        Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # standardize X
        return Z

    def PCA(self, X):
        # do principal component analysis
        Z = self.standardize(X.copy())
        C = Z.T @ Z / (X.shape[0] - 1) # covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(C)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        self.eig_vecs = eig_vecs
        X_reduced = Z @ eig_vecs # transformed data

        # dimension reduction
        eig_vals_cumsum = np.cumsum(eig_vals)
        n = 1
        while eig_vals_cumsum[n-1] / eig_vals_cumsum[-1] < self.explained_var:
           n += 1

        return X_reduced[:, :n]

    def fit(self, X, y):
        X_reduced = self.PCA(X)
        coefs, _, _, _ = np.linalg.lstsq(X_reduced, y, rcond=None)
        self.coefs = coefs

    def predict(self, X):
        Z = self.standardize(X.copy())
        X_reduced = (Z @ self.eig_vecs)[:, :len(self.coefs)]
        return X_reduced @ self.coefs


if __name__ == '__main__':
    angle = np.pi / 4
    r_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = np.random.rand(100, 2)
    X[:, 1] = np.random.normal(size=X.shape[0])
    X = X @ r_matrix
    y = X[:, 0]/2 + X[:, 1]/5

    # PCR fit and predict
    pcr = PrincipalComponentRegression()
    pcr.fit(X, y)
    y_hat = pcr.predict(X)
    mse = (np.sum((y - y_hat)**2) / X.shape[0])**0.5
    print(f'eigen vector:\n{pcr.eig_vecs}')
    print(f'error: {mse * 100}%, features: {len(pcr.coefs)}')

    # retrieve principal and secondary axes
    means = X.mean(axis=0)
    first_axis = np.vstack(([0, 0], pcr.eig_vecs[:, 0]*2))
    first_axis[:, 0] += means[0]
    first_axis[:, 1] += means[1]
    second_axis = np.vstack(([0, 0], pcr.eig_vecs[:, 1]*0.5))
    second_axis[:, 0] += means[0]
    second_axis[:, 1] += means[1]

    # plot
    plt.figure('X')
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')
    plt.plot(first_axis[:, 0], first_axis[:, 1], 'r')
    plt.plot(second_axis[:, 0], second_axis[:, 1], 'g')
    plt.show()
