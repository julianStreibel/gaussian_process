import numpy as np


class Gaussian_Process():
    def __init__(self, X: np.array, y: np.array, sigma: float = 1, l: float = 1):
        """Gaussian Process or Bayesian Kernel Regression model for mean and variance estimation."""
        self.sigma = sigma
        self.l = l
        self.X = X
        self.k = 1 if len(X.shape) == 1 else X.shape[1]
        self.y = y
        self.n = y.shape[0]
        self.init_kernel_matrix_K()

    def estimate(self, x_query):
        k_X = self.rbf_kernel(np.tile(x_query, (self.n, 1)),
                              self.X.reshape((-1, self.k)), axis=1)
        k_self = self.rbf_kernel(x_query, x_query)
        inv = np.linalg.pinv(self.K + np.eye(self.n) * self.sigma ** 2)

        mean_query = k_X @ inv @ self.y
        sigma_query = k_self + self.sigma ** 2 - k_X @ inv @ k_X

        return mean_query, sigma_query

    def init_kernel_matrix_K(self):
        self.K = np.eye(self.X.shape[0])
        for i in range(self.n):
            for j in range(i):
                k = self.rbf_kernel(self.X[i], self.X[j])
                self.K[i][j] = k
                self.K[j][i] = k

    def rbf_kernel(self, x_one: np.array, x_two: np.array, axis=None) -> float:
        return np.exp(- np.linalg.norm(x_one - x_two, axis=axis) / (2 * self.l ** 2))
