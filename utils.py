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
    newrange = [m,(m+n)/2] if not r else [(m+n)/2,n]
    bisect(func, m=newrange[0], n=newrange[1], tol=tol)



def get_opt_func(rate_matrices, distributions):
    """ Builds objective function for finding optimal eta value

    rate_matrices is list [ rate_matrix_1, rate_matrix_2, ...] of available rate matrices
    distributions is list [ prob_dist1, prob_dist2, ... ] to define projection 
        Pi(p) = argmin_{q in distributions} D(p||q)

    """

    n = len(rate_matrices[0])  # number of states

    def get_dS(W, p):
        """ Rate of Shannon entropy change, for rate matrix W and distribution p """

        """ We rewrite -sum_{i,j} p_i W_ji ln p_j as the "KL-like" expression
               1/tau sum_{i,j} p_i T_ji ln (p_i T_ji/p_j T_ji)
        where tau = -min_i W_ii is the fastest time scale in R and
        T_ji = delta_{ji} + tau W_ji is a conditional probability distribuiton. This 
        lets us indicate to cvxpy that -sum_{i,j} p_i W_ji ln p_j is convex in p.
        """

        tau  = -1/np.min(np.diag(W))
        T    = np.eye(n) + tau*W
        assert(np.all(T>=0))

        dS = 0.
        for i in range(n):
            for j in range(n):
                if i == j: 
                    continue
                if np.isclose(T[i,j],0):
                    continue
                dS += cp.kl_div( T[i,j] * p[j], T[i,j] * p[i]) + T[i,j] * p[j] - T[i,j] * p[i]
        return dS / tau


    def get_EF(W, p):
        """ EF rate, for rate matrix W and distribution p, defined as 
            sum_{i,j} p_i W_ji ln (W_ji/W_ji)
        """

        EF = 0.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.isclose(W[i,j],0) and np.isclose(W[j,i],0):
                    continue
                EF += W[i,j] * p[j] * np.log( W[i,j] / W[j,i] )
        return EF

    
    def f(eta):
        p          = cp.Variable( n, name='p')
        logQ_param = cp.Parameter(n, name='logQ')


        min_val = None

        print('-'*len(rate_matrices))

        for W in rate_matrices:
            sys.stdout.write('.')
            sys.stdout.flush()

            cons = [ p >= 0, sum(p) == 1 ]
            for q2 in distributions:
                assert(np.all(q2 > 0))
                cons.append( p @ (logQ_param-np.log(q2)) >= 0 )

            obj  = (1-eta)*get_dS(W, p) + get_EF(W, p) - eta*(W @ p)@logQ_param
            cons.append( obj <= -1e-6)
            prob = cp.Problem(cp.Minimize(0), cons)

            for q in distributions:
                logQ_param.value = np.log(q)

                prob.solve(solver=cp.ECOS, reltol=1e-12)
                if prob.status == 'infeasible':
                    continue

                else:
                    print('')
                    return False


        print('')
        return True

    return f

