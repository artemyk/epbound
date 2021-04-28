import numpy as np
import cvxpy as cp
import sys
from scipy.special import entr, rel_entr

def entropy(p):
    """ Compute Shannon entropy of probability vector p """
    return np.sum(entr(p))


def kl(p,q, normalize=False):
    """ Compute KL divergence between probability vectors p and q """
    if normalize:
        return np.sum(rel_entr(p/p.sum(),q/q.sum()))
    else:
        return np.sum(rel_entr(p,q))


def boltzmann(E, beta=1.0):
    """ Return Boltzmann distribution corresponding to energy function E and inverse temperature beta """
    p = np.exp(-beta*E)
    return p / p.sum()


def bisect(func, m, n, tol=1e-3):
    """ Perform bisection search for root """ 
    print('Search range: [%g-%g]' % (m,n))
    if (n-m) < tol:
        return
    r = func((m+n)/2)
    newrange = [m,(m+n)/2] if r < 0 else [(m+n)/2,n]
    bisect(func, m=newrange[0], n=newrange[1], tol=tol)



def get_opt_func(rate_matrices):
    """ Builds objective function for finding optimal eta value

    rate_matrices is list [ (rate_matrix_1, equilibrium_distribution_1), 
                            (rate_matrix_2, equilibrium_distribution_2), 
                            ...
                          ]

    """

    n = len(rate_matrices[0][0])  # number of states

    for ndx1, (L, pi) in enumerate(rate_matrices): # check for local detailed balance (LDB)
        if not np.allclose(L.dot(pi), 0):
            raise Exception('stationary distribution and rate matrix %d do not match' % ndx1)

    
    def get_dS(W, p):
        """ Rate of Shannon entropy change, for rate matrix W and distribution p """

        """ We rewrite sum_{i,j} p_i R_ji ln (p_i/p_j) as the "KL-like" expression
               1/tau sum_{i,j} p_i T_ji ln (p_i T_ji/p_j T_ji)
        where tau = -min_i R_ii is the fastest time scale in R and
        T_ji = delta_{ji} + tau R_ji is a conditional probability distribuiton. This 
        lets us indicate to cvxpy that sum_{i,j} p_i R_ji ln (p_i/p_j) is convex in p.
        """

        tau  = -1/np.min(np.diag(W))
        T    = np.eye(n) + tau*W
        assert(np.all(T>=0))

        dS = 0.
        for i in range(n):
            for j in range(n):
                if np.isclose(T[i,j],0):
                    continue
                dS += cp.kl_div( T[i,j] * p[j], T[i,j] * p[i]) + T[i,j] * p[j] - T[i,j] * p[i]
        return dS / tau

    
    def f(eta):
        p            = cp.Variable(n, name='p')
        logpi2_param = cp.Parameter(n, name='logpi2')

        cons = [ p >= 0, sum(p) == 1 ]
        for (L, pi) in rate_matrices:
            cons.append( p @ ( np.log(pi) - logpi2_param ) <= 0 )

        min_val = None

        print('-'*len(rate_matrices))

        for ndx1, (L, pi) in enumerate(rate_matrices):
            sys.stdout.write('.')
            sys.stdout.flush()
            dS   = get_dS(L, p)
            obj  = (1-eta)*dS + (L @ p)@np.log(pi) - eta*(L @ p)@logpi2_param

            prob = cp.Problem(cp.Minimize(obj), cons)

            for ndx2, (L2, pi2) in enumerate(rate_matrices):
                if ndx1 == ndx2:
                    continue

                logpi2_param.value = np.log(pi2)

                prob.solve(solver=cp.ECOS)
                
                if prob.status == 'infeasible':
                    continue

                if np.isnan(obj.value):
                    print(pi2, pi, W)
                    raise Exception('nan returned', ndx1, ndx2)

                if min_val is None or obj.value < min_val:
                    min_val = obj.value

                if min_val < -1e-8:
                    print('')
                    return min_val

        print('')
        return min_val

    return f

