import numpy as np
import cvxpy as cp
import sys

def entropy(p):
    return -p.dot(np.log(p))


def kl(p2,q2, normalize=False):
    p = p2.copy()
    q = q2.copy()
    if normalize:
        p /= p.sum()
        q /= q.sum()    
    return sum([p[i]*np.log(p[i]/q[i]) for i in range(len(p)) if not np.isclose(p[i], 0)])


def boltzmann(E):
    p = np.exp(-E)
    return p / p.sum()


def bisect(func, m, n, tol=1e-3):
    print(m,n)
    if (n-m) < tol:
        return
    r = func((m+n)/2)
    newrange = [m,(m+n)/2] if r < 0 else [(m+n)/2,n]
    bisect(func, m=newrange[0], n=newrange[1], tol=tol)



def get_opt_func(rate_matrices):
    
    n = len(rate_matrices[0][0])
    
    def get_dS(W, p):
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

