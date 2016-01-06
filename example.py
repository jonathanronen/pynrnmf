import numpy as np
from pynrnmf import NRNMF


F = np.array([[1,10,2,4], [1,2,1,1], [20,5,3,15], [10,11,2,3], [40,2,21,1], [1,2,1,10]])

model = NRNMF(alpha=100, init='random', n_inits=100)
U, V = model.fit_transform(F)