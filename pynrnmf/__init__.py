import numpy as np
from warnings import warn
from scipy import sparse, linalg
from joblib import Parallel, delayed

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
    def __init__(self, k=None, W=None, alpha=100, init='random', n_inits=1, tol=1e-3, max_iter=1000, n_jobs=1, parallel_backend='multiprocessing'):
        """
            k:          number of components
            W:          Adjacency matrix of the nearest neighboor matrix
            alpha:      regularization parameter
            init:       'random' initializes to a random W,H
            n_inits:    number of runs to make with different random inits (in order to avoid being stuck in local minima)
            n_jobs:     number of parallel jobs to run, when n_inits > 1
            tol:        stopping criteria
            max_iter:   stopping criteria
        """
        self.k = k
        self.W = W
        if W is not None:
            self.D = sparse.coo_matrix(np.diagflat(W.sum(axis=1)))
            self.L = self.D - self.W
        self.alpha = alpha
        self.init = init
        self.n_inits = n_inits
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    def _init(self, X):
        if self.init == 'random':
            U = np.random.random((X.shape[0], self.k))
            V = np.random.random((X.shape[1], self.k))
        else:
            raise NotImplementedError("Don't know the '{}' init method. Must be 'random'".format(self.init))
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
        return linalg.norm(X - U.dot(V.T)) + self.alpha * np.trace(V.T * self.L * V)

    def _fit(self, X):
        U, V = self._init(X)
        conv = False
        for x in range(self.max_iter):
            Un, Vn = self._update(U, V, X, self.alpha)
            e = linalg.norm(U-Un)
            U, V = Un, Vn
            if e < self.tol:
                conv = True
                break
        return {
            'conv': conv,
            'e': e,
            'U': U,
            'V': V
        }

    def fit_transform(self, X):
        """
        Fits the model to the data matrix X, and returns the decomposition U, V.
        """
        if self.k is None:
            self.k = X.shape[1]
        if self.W is None:
            self.W = np.eye(X.shape[1])
            self.D = np.eye(X.shape[1])
            self.L = self.D - self.W

        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(delayed(self._fit)(X) for x in range(self.n_inits))
        best_results = {"e": np.inf, "U": None, "V": None}
        for r in results:
            if r['e'] < best_results['e']:
                best_results = r
        if not best_results['conv']:
            warn("Did not converge after {} iterations. Error is {}. Try increasing `max_iter`.".format(self.max_iter, best_results['e']))
        return best_results["U"], best_results["V"]
