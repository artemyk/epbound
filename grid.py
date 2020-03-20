import sys

if not sys.version_info >= (3,5,0):
    raise Exception('Python 3.5 or higher required')
    
import numpy as np
import time
from utils import *

np.set_printoptions(precision=2)

def get_ixs(width, height):
    ixs = {}
    for i in range(width):
        for j in range(height):
            ixs[(i,j)] = len(ixs)
    rev_ixs = { v:k for k,v in ixs.items() }
    return ixs, rev_ixs

height = 4
width  = 4

ixs, rev_ixs = get_ixs(width, height)

def build_rate_matrix(pi):
    L = np.zeros( (len(ixs), len(ixs)) )
    for (x1,y1), ix1 in ixs.items():
        for (x2,y2), ix2 in ixs.items():
            if ix1 == ix2: 
                continue
            if np.abs(x1 - x2) > 1.5 or np.abs(y1 - y2) > 1.5:
                continue
            L[ix2, ix1] = pi[ix2]
    L = L - np.diag(np.sum(L, axis=0))
    return L

rate_matrices = []
for k, v in ixs.items():
    E = np.zeros(len(ixs))
    E[v] = 1.0
    pi = boltzmann(-E) 
    L = build_rate_matrix(pi)
    rate_matrices.append( (L, pi) )


print('Bisecting for optimal eta')
start_time = time.time()
v = get_opt_func(rate_matrices)
bisect(v, 0, 1, tol=1e-2) 
print('Finished in %0.3f s' % (time.time() - start_time) )