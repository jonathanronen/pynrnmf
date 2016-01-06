"""
This example script takes some random data points and performs NMF based on a two-component adjacency matrix.
Assumes two first points belong to first component, two other points to the second network component.
Then after factorization, we see that with higher regularization, points in the same network component are shrunken closer together
in the NMF representation.

@jonathanronen 2016
"""

import numpy as np
from pynrnmf import NRNMF

# Data matrix has 4 6-dimensional points
F = np.random.random((6,4))

# Adjacency matrix has two network modules (two connected components)
W = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])

model = NRNMF(k=2, W=W, alpha=10000, init='random', n_inits=10, max_iter=20000)
U, V = model.fit_transform(F)

model_lowalpha = NRNMF(k=2, W=W, alpha=100, init='random', n_inits=10, max_iter=10000)
U_la, V_la = model_lowalpha.fit_transform(F)

model_noreg = NRNMF(k=2, W=W, alpha=0, init='random', n_inits=10, max_iter=5000)
U_nr, V_nr = model_noreg.fit_transform(F)

print("Reconstruction error with high regularization: {}".format(np.linalg.norm(F-U.dot(V.T))))
print("Reconstruction error with low regularization: {}".format(np.linalg.norm(F-U_la.dot(V_la.T))))
print("Reconstruction error with no regularization: {}".format(np.linalg.norm(F-U_nr.dot(V_nr.T))))

mean_dist_same_high = (np.linalg.norm(V[0,:] - V[1,:]) + np.linalg.norm(V[2,:] - V[3,:])) / 2
mean_dist_same_low = (np.linalg.norm(V_la[0,:] - V_la[1,:]) + np.linalg.norm(V_la[2,:] - V_la[3,:])) / 2
mean_dist_same_no = (np.linalg.norm(V_nr[0,:] - V_nr[1,:]) + np.linalg.norm(V_nr[2,:] - V_la[3,:])) / 2
print("Mean distance of NMF representations within same network module, high regularization: {}".format(mean_dist_same_high))
print("Mean distance of NMF representations within same network module, low regularization: {}".format(mean_dist_same_low))
print("Mean distance of NMF representations within same network module, no regularization: {}".format(mean_dist_same_no))

mean_dist_diff_high = (np.linalg.norm(V[0,:] - V[2,:]) + np.linalg.norm(V[0,:] - V[3,:]) + np.linalg.norm(V[1,:] - V[2,:])) / 3
mean_dist_diff_low = (np.linalg.norm(V_la[0,:] - V_la[2,:]) + np.linalg.norm(V_la[0,:] - V_la[3,:]) + np.linalg.norm(V_la[1,:] - V_la[2,:])) / 3
mean_dist_diff_no = (np.linalg.norm(V_nr[0,:] - V_nr[2,:]) + np.linalg.norm(V_nr[0,:] - V_nr[3,:]) + np.linalg.norm(V_nr[1,:] - V_nr[2,:])) / 3
print("Mean distance of NMF representations from different network module, high regularization: {}".format(mean_dist_diff_high))
print("Mean distance of NMF representations from different network module, low regularization: {}".format(mean_dist_diff_low))
print("Mean distance of NMF representations from different network module, no regularization: {}".format(mean_dist_diff_no))

