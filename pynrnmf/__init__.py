import numpy as np
from scipy import sparse
from warnings import warn

class NRNMF:
    """
    Performs Network-regularized Non-Negative matrix factorization.

    Implements algorithm from [1].

    [1]: Non-negative Matrix Factorization on Manifold, Cai et al, ICDM 2008
         http://dx.doi.org/10.1109/ICDM.2008.57
    """
    def __init__(self, k=None, A=None, alpha=100, init='random', n_inits=100, tol=1e-3, max_iter=100):
        """
            k:          number of components
            A:          Adjacency matrix of the nearest neighboor matrix
            alpha:      regularization parameter
            init:       'random' initializes to a random W,H
            n_inits:    number of runs to make with different random inits (in order to avoid being stuck in local minima)
            tol:        stopping criteria
            max_iter:   stopping criteria
        """
        self.k = k
        self.A = A
        if A:
            self.D = sparse.coo_matrix(np.diagflat(A.sum(axis=1)))
        self.alpha = alpha
        self.init = init
        self.n_inits = n_inits
        self.tol = tol
        self.max_iter = max_iter

    def _init(self, X):
        if self.init == 'random':
            W = np.random.random((X.shape[0], self.k))
            H = np.random.random((self.k, X.shape[1]))
        else:
            raise NotImplementedError("Don't know the '{}' init method. Must be 'random'".format(self.init))
        return W, H

    def _update(self, W, H, X, alpha):
        XV = X.dot(H.T)
        UVtV = W.dot(H.T.dot(H))

        XtUplWV = X.T.dot(W) + alpha * self.A.dot(H)
        VUtplDV = H.dot(W.T.dot(W)) + alpha * self.D.dot(H)

        W = np.multiply(W, np.divide(XV, UVtV))
        H = np.multiply(H, np.divide(XtUplWV, VUtplDV))
        return W, H

    def _error(self, W, H, X):
        return np.linalg.norm(X - W.dot(H.T))

    def fit_transform(self, X):
        """
        Fits the model to the data matrix X, and returns the decomposition W, H.
        """
        if self.k is None:
            self.k = X.shape[1]
        if self.A is None:
            self.A = np.eye(X.shape[1])
            self.D = np.eye(X.shape[1])

        W, H = self._init(X)
        for i in range(self.max_iter):
            W, H = self._update(W, H, X, self.alpha)
            if self._error(W, H, X) < self.tol:
                return W, H
        warn("Did not converge. Error is {} after {} iterations.".format(self._error(W, H, X), i+1))
        return W, H
