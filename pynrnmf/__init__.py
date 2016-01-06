import numpy as np
from scipy import sparse
from warnings import warn

class NRNMF:
    """
    Performs Network-regularized Non-Negative matrix factorization.

    Implements algorithm from [1].
    Optimize objective function
        O = ||X - U V^T||_F^2 + lambda Tr(V^T L V)

    where L is the graph laplacian of the graph defined by the edge weight matrix W
    (L = D - W, where D is a diagonal whose entries are row/col sums of W).

    The algorithm is an iteration defined by the update steps

        Uij <- Uij * (XV)ij / (U V^T V)ij
        Vij <- Vij * (X^T U + lambda W V)ij / (V U^T U + lambda D V)ij

    This factorizes an X (MxN) matrix into U (Mxk) and V (Nxk) matrices.

    [1]: Non-negative Matrix Factorization on Manifold, Cai et al, ICDM 2008
         http://dx.doi.org/10.1109/ICDM.2008.57
    """
    def __init__(self, k=None, W=None, alpha=100, init='random', n_inits=100, tol=1e-3, max_iter=1000):
        """
            k:          number of components
            W:          Adjacency matrix of the nearest neighboor matrix
            alpha:      regularization parameter
            init:       'random' initializes to a random W,H
            n_inits:    number of runs to make with different random inits (in order to avoid being stuck in local minima)
            tol:        stopping criteria
            max_iter:   stopping criteria
        """
        self.k = k
        self.W = W
        if W:
            self.D = sparse.coo_matrix(np.diagflat(W.sum(axis=1)))
        self.alpha = alpha
        self.init = init
        self.n_inits = n_inits
        self.tol = tol
        self.max_iter = max_iter

    def _init(self, X):
        if self.init == 'random':
            U = np.random.random((X.shape[0], self.k))
            V = np.random.random((self.k, X.shape[1]))
        else:
            raise NotImplementedError("Don't know the '{}' init method. Must be 'random'".format(self.init))
        # return W, H
        return U,V

    def _update(self, U, V, X, alpha):
        XV = X.dot(V)
        UVtV = U.dot(V.T.dot(V))

        XtUpaWV = X.T.dot(U) + alpha * self.W.dot(V)
        VUtUpaDV = V.dot(U.T.dot(U)) + alpha * self.D.dot(V)

        U = np.multiply(U, np.divide(XV, UVtV))
        V = np.multiply(V, np.divide(XtUpaWV, VUtUpaDV))
        return U, V


    def _error(self, U, V, X):
        return np.linalg.norm(X - U.dot(V.T))

    def fit_transform(self, X):
        """
        Fits the model to the data matrix X, and returns the decomposition W, H.
        """
        if self.k is None:
            self.k = X.shape[1]
        if self.W is None:
            self.W = np.eye(X.shape[1])
            self.D = np.eye(X.shape[1])

        U, V = self._init(X)
        for i in range(self.max_iter):
            U, V = self._update(U, V, X, self.alpha)
            if self._error(U, V, X) < self.tol:
                return U, V
        warn("Did not converge. Error is {} after {} iterations.".format(self._error(U, V, X), i+1))
        return U, V
