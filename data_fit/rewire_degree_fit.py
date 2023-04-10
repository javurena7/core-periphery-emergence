import numpy as np
import sys
sys.path.append('..')
import simulation_model as am
import sympy as sym
from scipy import optimize
from scipy.special import expit
from math import factorial
from itertools import product


class RewireFit(object):
    def __init__(self, x_evol, n_evol):
        """
        Main function for fitting c, sa, sb from data.
        Input:
            x_evol: dictionary of evolution steps. Of the form {t: [linktype, tgt_k, tgt_cnt]}, where t is the timestep, linktype one of 'aa', 'ab', 'ba', 'bb', tgt_k the degree of the accepted candidate/target node, and tgt_cnt the number of nodes of degree tgt_k in the group of the target node.
            n_evol: dictionary of group-level denomitors of the llik function at the corresponding time step. Of the form {t: [Pa_tot, Pb_tot, Ua_tot, Ub_tot]}, where:
                Pa_tot is sum(n_k*k), the sum of degree*(times the degree happens in group a)
                Pb_tot is sum(n_k*k), the sum of degree*(times the degree happens in goup b)
                Ua_tot is sum(n_k) the times that each degree happens in group a
                Ub_tot is sum(n_k) the times that each degree happens in group b
        """
        self.x_evol = x_evol
        self.n_evol = n_evol

    def loglik(self, theta): #c, sa, sb, x, n):
        sa, sb, c = theta
        llik = 0
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1: # and (xt['aa'] or xt['ab'] or xt['bb'] or xt['ba']):
                llik -= _lik_step_t_light(c, sa, sb, nt, xt)
        return llik

    def grad_loglik(self, theta): #c, sa, sb, x, n):
        grad = np.zeros(3)
        sa, sb, c = theta
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1:
                grad -= _gradlik_step_t_probs(c, sa, sb, nt, xt)
        return grad

    def hess_loglik(self, theta): #c, sa, sb, x, n):
        sa, sb, c = theta
        hess = np.zeros((3, 3))
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1:
                hess -= _hesslik_step_t_probs(c, sa, sb, nt, xt)
        return hess

    def solve(self, x0=[.5, .5, .5], method='trust-constr', bounds=()):
        """
        Fit data. Returns array with sa, sb, c
        Input:
            x0: initial conditions for three paramters, list of three values
            method: method for scipy.optimize.minimize
            bounds: array with lower, and upper bounds for the three paramters
                fomrmat: ([lsa, lsb, lc], [usa, usb, uc]) for lower and upper bounds
        """
        bounds = ([.05]*3, [.95]*3) if not bounds else bounds
        bounds = optimize.Bounds(*bounds)
        opt = optimize.minimize(self.loglik, x0, method=method, jac=self.grad_loglik, hess=self.hess_loglik, bounds=bounds)
        self.opt = opt
        return opt.x

    def solve_randx0(self, n_x0=10, method='trust-constr'):
        opt = None
        fun = np.inf
        for _ in range(n_x0):
            x0 = np.random.uniform(.1, .9, 3)
            x = self.solve(x0, method)
            if self.opt.fun < fun:
                fun = self.opt.fun
                opt = self.opt
        return opt.x

    def loglik_c0(self, theta): #c, sa, sb, x, n):
        sa, sb = theta
        llik = 0
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1: # and (xt['aa'] or xt['ab'] or xt['bb'] or xt['ba']):
                llik -= _lik_step_t_light(0, sa, sb, nt, xt)
        return llik

    def grad_loglik_c0(self, theta): #c, sa, sb, x, n):
        grad = np.zeros(3)
        sa, sb = theta
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1:
                grad -= _gradlik_step_t_probs(0, sa, sb, nt, xt)
        grad = grad[:2]
        return grad

    def hess_loglik_c0(self, theta): #c, sa, sb, x, n):
        sa, sb = theta
        hess = np.zeros((3, 3))
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1:
                hess -= _hesslik_step_t_probs(0, sa, sb, nt, xt)
        hess = hess[:2, :2]
        return hess

    def solve_c0(self, x0=[.5, .5], method='trust-constr'):
        """
        Fit sa, sb assuming that there is no preferential attachment (c=0)
        """
        bounds = optimize.Bounds([0.05, 0.05], [.95, .95])
        opt = optimize.minimize(self.loglik_c0, x0, method=method, jac=self.grad_loglik_c0, hess=self.hess_loglik_c0, bounds=bounds)
        self.opt_c0 = opt
        return opt.x




def get_dataset(sa, sb, na, c, N, n_iter=2000, m=10, p0=[[.02, .02], [.02, .02]], light=True):
    """
    Get sample dataset by running simulations
        N: number of nodes
        p0: local densities for network; matrix size 2*2
    """
    if light:
        deg_based = 'light'
    else:
        deg_based = True
    x, n = am.run_rewiring(N, na, c, bias=[sa, sb], p0=p0, n_iter=n_iter, m_dist=['poisson', m], deg_based=deg_based)
    return x, n

def get_dataset_varm(sa, sb, na, c, N, m_dist=['poisson', 50], m0=10, em=True):
    Paa, Pbb, counts = am.run_growing_varying_m(N, na, c, [sa, sb], m_dist, m0=m0, ret_counts=True)
    if em:
        obs_counts = np.array(counts)[:, [0, 1, 3, 4]]
        return Paa, Pbb, obs_counts, counts
    else:
        return Paa, Pbb, counts

def _get_h(xt, sa, sb):
    if xt.get('aa'):
        return 'a', 'a', sa, (1-sa)
    elif xt.get('ab'):
        return 'a', 'b',sa, (1-sa)
    elif xt.get('ba'):
        return 'b', 'a', (1-sb), sb
    elif  xt.get('bb'):
        return 'b', 'b', (1-sb), sb
    else:
        return '', '', 0, 0


def _get_h_light(xt, sa, sb):
    if xt[0] == 'aa':
        return 'a', 'a', sa, (1-sa)
    elif xt[0] == 'ab':
        return 'a', 'b',sa, (1-sa)
    elif xt[0] == 'ba':
        return 'b', 'a', (1-sb), sb
    elif xt[0] == 'bb':
        return 'b', 'b', (1-sb), sb
    else:
        return '', '', 0, 0


def _lik_step_t_light(c, sa, sb, nt, xt):
    src, tgt, ha, hb = _get_h_light(xt, sa, sb)
    if not src:
        return 0

    Pa_tot = nt[0] #'pa'] #sum([n*k for k, n in nt['na'].items()])
    Pb_tot = nt[1] #'pb'] #sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = nt[2] #'ua'] #sum(nt['na'].values())
    Ub_tot = nt[3] #'ub'] #sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    pi_a = ha * (c*Pa_tot/P_denm + (1-c)*Ua_tot/U_denm)
    pi_b = hb * (c*Pb_tot/P_denm + (1-c)*Ub_tot/U_denm)

    # Step likelihood
    tgt_deg = xt[1]
    n_k = xt[2]
    p_k = c * n_k * tgt_deg / P_denm + (1 - c)* n_k / U_denm

    if tgt == 'a':
        p_k = ha * p_k
    else:
        p_k = hb * p_k

    # Full likelihood
    llik = np.log(p_k) if p_k > 0 else 0
    llik += - 1 * np.log(pi_a + pi_b) if pi_a + pi_b > 0 else 0

    return llik


def _lik_step_t_probs(c, sa, sb, nt, xt):
    #src, ha, hb = _get_h(xt, sa, sb)
    #if not src:
    #    return 0
    saa = sa
    sab = 1 - sa
    sbb = sb
    sba = 1 - sb

    Pa_tot = sum([n*k for k, n in nt['na'].items()])
    Pb_tot = sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = sum(nt['na'].values())
    Ub_tot = sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    #P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    #P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    #x_tot = 1 + sum(xt[src+'a'].values()) + sum(xt[src+'b'].values())

    llik_a, llik_b = 0, 0 #lliks for when sources are a and b
    Pa_tot = 0 #For sum of probs when src is a
    Pb_tot = 0 #For sum of probs when src is b

    #Prob of going from a to a
    for k, xk in xt['aa'].items():
        na_k = nt['na'].get(k, 0)
        pia_k = c * na_k*k/P_denm if P_denm > 0 else 0
        pia_k += (1-c)*(na_k/U_denm) if U_denm > 0 else 0
        Pa_tot += pia_k*saa
        llik_a += xk*(np.log(pia_k) + np.log(saa)) if pia_k > 0 else 0
        #llik_a *= (pia_k*saa)**xk if pia_k > 0 else 1

    #Prob of going from a to b
    for k, xk in xt['ab'].items():
        nb_k = nt['nb'].get(k, 0)
        pib_k = c * nb_k*k/P_denm if P_denm > 0 else 0
        pib_k += (1-c)*(nb_k/U_denm) if U_denm > 0 else 0
        Pa_tot += pib_k*sab
        llik_a += xk*(np.log(pib_k) + np.log(sab)) if pib_k > 0 else 0
        #llik_a *= (pib_k*sab)**xk if pib_k > 0 else 1

   #Prob of going from b to a
    for k, xk in xt['ba'].items():
        na_k = nt['na'].get(k, 0)
        pia_k = c * na_k*k/P_denm if P_denm > 0 else 0
        pia_k += (1-c)*(na_k/U_denm) if U_denm > 0 else 0
        Pb_tot += pia_k*sba
        llik_b += xk*(np.log(pia_k) + np.log(sba)) if pia_k > 0 else 0
        #llik_b *= (pia_k*sba)**xk if pia_k > 0 else 1

    #Prob of going from b to b
    for k, xk in xt['bb'].items():
        nb_k = nt['nb'].get(k, 0)
        pib_k = c * nb_k*k/P_denm if P_denm > 0 else 0
        pib_k += (1-c)*(nb_k/U_denm) if U_denm > 0 else 0
        Pb_tot += pib_k*sbb
        llik_b += xk*(np.log(pib_k) + np.log(sbb)) if pib_k > 0 else 0
        #llik_b *= (pib_k*sbb)**xk if pib_k > 0 else 1

    xa_tot = 1 + sum(xt['aa'].values()) + sum(xt['ab'].values())
    llik_a -=  xa_tot * np.log(Pa_tot) if Pa_tot > 0 else 0
    #llik_a += np.log(factorial(xa_tot - 1)) if xa_tot > 1 else 0
    #for a in xt['aa'].values():
    #    llik_a -= np.log(factorial(a)) if a > 0 else 0
    #for a in xt['ab'].values():
    #    llik_a -= np.log(factorial(a)) if a > 0 else 0
    #llik_a *=  Pa_tot**(-xa_tot) if Pa_tot > 0 else 1

    xb_tot = 1 + sum(xt['ba'].values()) + sum(xt['bb'].values())
    llik_b -=  xb_tot * np.log(Pb_tot) if Pb_tot > 0 else 0
    #llik_b += np.log(factorial(xb_tot - 1)) if xb_tot > 2 else 0
    #for b in xt['ba'].values():
    #    llik_b -= np.log(factorial(b)) if b > 0 else 0
    #for b in xt['bb'].values():
    #    llik_b -= np.log(factorial(b)) if b > 0 else 0
    #llik_b *= Pb_tot**(-xb_tot) if Pb_tot > 0 else 1

    lik = llik_a + llik_b

    return -lik


def _gradlik_step_t_probs(c, sa, sb, nt, xt):
    src, tgt, ha, hb = _get_h_light(xt, sa, sb)
    if not src:
        return 0
    L = np.zeros(3)

    Pa_tot = nt[0] #'pa'] #sum([n*k for k, n in nt['na'].items()])
    Pb_tot = nt[1] #'pb'] #sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = nt[2] #'ua'] #sum(nt['na'].values())
    Ub_tot = nt[3] #'ub'] #sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tota = 1 if xt[0] in ['aa', 'ab'] else 0
    x_totb = 1 if xt[0] in ['bb', 'ba'] else 0

    #llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0
    Ub = Ub_tot / U_denm
    Ua = Ua_tot / U_denm
    Dca = Pa_tot / P_denm - Ua
    Dcb = Pb_tot / P_denm - Ub
    DUa = Ua - Ub
    DUb = Ub - Ua
    DL_dsa_num = c*(Dca - Dcb) + DUa
    DL_dsa = DL_dsa_num / (sa * DL_dsa_num + c*Dcb + Ub)

    #L[0] = 1/sa * xt.get('aa', 0) - (1/(1-sa))*xt.get('ab', 0) - x_tota*DL_dsa
    L[0] = 0#- x_tota * DL_dsa
    if xt[0] == 'aa':
        L[0] += 1 / sa - x_tota * DL_dsa
    elif xt[0] == 'ab':
        L[0] += - 1 / (1-sa) - x_tota * DL_dsa

    DL_dsb_num = c*(Dcb - Dca) + DUb
    DL_dsb = DL_dsb_num / (sb * DL_dsb_num + c*Dca + Ua)
    #L[1] = 1/sb * xt.get('bb', 0) - (1/(1-sb))* xt.get('ba', 0) - x_totb*DL_dsb
    L[1] = 0 #- x_totb * DL_dsb
    if xt[0] == 'bb':
        L[1] += 1/sb - x_totb * DL_dsb
    elif xt[0] == 'ba':
        L[1] += - 1 / (1 - sb) - x_totb * DL_dsb

    x_tot = 1 #+ xt.get(src+'a', 0) + xt.get(src+'b', 0)
    if src == 'a':
        num = sa*(Dca - Dcb) + Dcb
        den = c * num + sa*DUa + Ub
    if src == 'b':
        num = sb*(Dcb - Dca) + Dca
        den = c * num + sb*DUb + Ua
    L[2] += -x_tot*num/den


    tgt_deg = xt[1]
    n_k = xt[2]
    p_k = n_k * tgt_deg / P_denm
    u_k = n_k / U_denm
    Dc_k = p_k - u_k
    L[2] += Dc_k / (c*Dc_k + u_k) if c*Dc_k + u_k > 0 else 0

    return L

def _hesslik_step_t_probs(c, sa, sb, nt, xt):
    src, tgt, ha, hb = _get_h_light(xt, sa, sb)
    if not src:
        return 0
    H = np.zeros((3, 3))

    Pa_tot = nt[0] #'pa'] #sum([n*k for k, n in nt['na'].items()])
    Pb_tot = nt[1] #'pb'] #sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = nt[2] #'ua'] #sum(nt['na'].values())
    Ub_tot = nt[3] #'ub'] #sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tota = 1 if xt[0] in ['aa', 'ab'] else 0
    x_totb = 1 if xt[0] in ['bb', 'ba'] else 0

    #llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0
    Ub = Ub_tot / U_denm
    Ua = Ua_tot / U_denm
    Dca = Pa_tot / P_denm - Ua
    Dcb = Pb_tot / P_denm - Ub
    DUa = Ua - Ub
    DUb = Ub - Ua

    DL_dsa_num = (c*(Dca - Dcb) + DUa)
    DL_dsa = DL_dsa_num**2 / (sa * DL_dsa_num + c*Dcb + Ub)**2

    #H[0, 0] = -1/sa**2 * sum(xt['aa'].values()) - (1/(1-sa)**2)*sum(xt['ab'].values()) + x_tota*DL_dsa
    if xt[0] == 'aa':
        H[0, 0] = - (1 / sa)**2 - x_tota * DL_dsa
    elif xt[0] == 'ab':
        H[0, 0] = - (1 /(1-sa))**2 - x_tota * DL_dsa
    DL_dsb_num = c*(Dcb - Dca) + DUb
    DL_dsb = DL_dsb_num**2 / (sb * DL_dsb_num + c*Dca + Ua)**2

    #H[1, 1] = 1/sb * sum(xt['bb'].values()) - (1/(1-sb))*sum(xt['ba'].values()) - x_totb*DL_dsb
    if xt[0] == 'bb':
       H[1, 1] = -(1/sb)**2 - x_totb * DL_dsb
    elif xt[0] == 'ba':
       H[1, 1] = -(1/(1-sb))**2 - x_totb * DL_dsb

    H[0, 2] = ((Dca-Dcb)*Ub - Dcb*DUa) / (c*(Dcb +sa*(Dca-Dcb))+sa*DUa+Ub)**2
    H[2, 0] = H[0, 2]

    H[1, 2] = ((Dcb-Dca)*Ua - Dca*DUb) / (c*(Dca +sb*(Dcb-Dca))+sb*DUb+Ua)**2
    H[2, 1] = H[1, 2]

    x_tot = 1 #1 + sum(xt[src+'a'].values()) + sum(xt[src+'b'].values())
    if src == 'a':
        num = sa*(Dca - Dcb) + Dcb
        den = c * num + sa*DUa + Ub
    if src == 'b':
        num = sb*(Dcb - Dca) + Dca
        den = c * num + sb*DUb + Ua
    H[2, 2] += x_tot*(num**2)/den**2

    tgt_deg = xt[1]
    n_k = xt[2]
    p_k = n_k * tgt_deg / P_denm
    u_k = n_k / U_denm
    Dc_k = p_k - u_k
    H[2, 2] -= Dc_k / (c*Dc_k + u_k)**2 if c*Dc_k + u_k > 0 else 0

    return H

def c_loglik(sa, sb, x, n):
    vals = [loglik(cv, sa, sb, x, n) for cv in np.linspace(0, 1, 50)]
    return vals

def sa_loglik(c, sb, x, n):
    vals = [loglik(c, sa, sb, x, n) for sa in np.linspace(0, 1, 50)]
    return vals

def full_lik_func(x_evol, n_evol, n_size=50):
    pr = np.linspace(0.01, .99, n_size)
    fun = np.zeros((n_size, n_size, n_size))
    for (i, sa), (j, sb), (k, c) in product(enumerate(pr), enumerate(pr), enumerate(pr)):
        for (t, xt), nt in zip(x_evol.items(), n_evol.values()):
            if t > 1:
                fun[i, j, k] -= _lik_step_t_light(c, sa, sb, nt, xt)
    return fun



class RewireLinkFit(object):
    """
    Testing code for visualizing llik function over larger time steps
    """
    def __init__(self, x, n):
        self.x = x
        self.n = n
        self.sols = {}

    def solve(self, x0=[.5, .5, .5]):
        bounds = optimize.Bounds([.05, .05, .05], [.95, .95, .95])
        for (t, xt), nt in zip(self.x.items(), self.n.values()):
            def loglik(theta):
                sa, sb, c = theta
                return -_lik_step_t_light(c, sa, sb, nt, xt)

            def grad(theta):
                sa, sb, c = theta
                return -_gradlik_step_t_probs(c, sa, sb, nt, xt)

            def hess(theta):
                sa, sb, c = theta
                return -_hesslik_step_t_probs(c, sa, sb, nt, xt)

            opt = optimize.minimize(loglik, x0, method='trust-constr',
                    jac=grad, hess=hess, bounds=bounds)
            if xt[0][0] == 'a':
                self.sols[t] = ['sa', opt.x[0], opt.x[2]]
            else:
                self.sols[t] = ['sb', opt.x[1], opt.x[2]]
            print(f'{t}: {self.sols[t]}')

    def grids(self, n_size=10):
        gr = np.linspace(0.05, 0.95, n_size)
        funs = {}
        for (t, xt), nt in zip(self.x.items(), self.n.values()):
            print(t)
            vals = np.zeros((n_size, n_size))
            if xt[0][0] == 'a':
                for (i, sa), (j, c) in product(enumerate(gr), enumerate(gr)):
                    vals[i, j] -= _lik_step_t_light(c, sa, 0.5, nt, xt)
                funs[t] = ('sa', vals)
            else:
                for (i, sb), (j, c) in product(enumerate(gr), enumerate(gr)):
                    vals[i, j] -= _lik_step_t_light(c, 0.5, sb, nt, xt)
                funs[t] = ('sb', vals)
        self.funs = funs


def grow_gradloglikbase_jointmulti(sa, sb, c, Paa, Pbb, cnt, na):
    """
    Each vector contains the log derivaties of Maa, Mab, Man, Mba, Mbb, Mbn
    """
    Dsa = [1/sa, 1/(sa-1), (2 - 4*na + 2*c*(-1 + 2*na - Paa + Pbb))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa)), 0, 0, 0]
    Dsb = [0, 0, 0, 1/(sb-1), 1/sb, (2 - 4*na + 2*c*(-1 + 2*na - Paa + Pbb))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(na + sb - 2*na*sb))]
    Dc = [0, 0, 0, 0, 0, 0]
    Dc[0] = (1 - 2*na + Paa - Pbb)/(2*na + c*(1 - 2*na + Paa - Pbb))
    Dc[1] = (-1 + 2*na - Paa + Pbb)/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))
    Dc[2] = ((-1 + 2*na - Paa + Pbb)*(-1 + 2*sa))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))
    Dc[3] =(1 - 2*na + Paa - Pbb)/(2*na + c*(1 - 2*na + Paa - Pbb))
    Dc[4] =(-1 + 2*na - Paa + Pbb)/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))
    Dc[5] =((-1 + 2*na - Paa + Pbb)*(-1 + 2*sb))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(-1 + na + sb - 2*na*sb))

    x1, x2, x4, x5 = cnt
    x3 = x1 + x2 + 1
    x6 = x4 + x5 + 1
    cnts = [x1, x2, -x3, x4, x5, -x6]

    Lsa = sum([x*p for x, p in zip(cnts, Dsa)]) #real likelihood
    Lsb = sum([x*p for x, p in zip(cnts, Dsb)])
    Lc = sum([x*p for x, p in zip(cnts, Dc)])
    L = np.array([Lsa, Lsb, Lc])
    return L


def grow_hessloglik_base(sa, sb, c, Paa, Pbb, cnt, na):
    """
    Hessian Matrix for the growth model
    """
    h = np.zeros((3,3))
    x1, x2, x4, x5 = cnt
    x3 = (x1 + x2 + 1)
    x6 = (x4 + x5 + 1)

    #D[dmaasa*x1 + dmabsa*x2 - dmansa*x3, sa] // Simplify // FortranForm
    h[0,0] =-(x1/sa**2) - x2/(-1 + sa)**2 + (4*(1 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))**2
    h[0,1] = 0
    #D[-x3*dmansa, c] // Simplify // FortranForm
    h[0,2] =(2*(-1 + 2*na - Paa + Pbb)*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))**2

    h[1,0] = 0
    #D[dmbbsb*x5 + dmbasb*x4 - dmbnsb*x6, sb] // Simplify // FortranForm
    h[1,1] = -(x4/sb**2) - x5/(-1 + sb)**2 + (4*(1 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2*x6)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(na + sb - 2*na*sb))**2
    #D[-x6*dmbnsb, c] // Simplify // FortranForm
    h[1,2] = (-2*(-1 + 2*na - Paa + Pbb)*x6)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(na + sb - 2*na*sb))**2

    #D[-x3*dmanc, sa] // Simplify // FortranForm
    h[2,0] =(2*(-1 + 2*na - Paa + Pbb)*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) +2*(-1 + na + sa - 2*na*sa))**2
    #D[-x6*dmbnc, sb] // Simplify // FortranForm
    h[2,1] = (-2*(-1 + 2*na - Paa + Pbb)*x6)/(-2 + (2 - 2*na + c*(-1 + 2*na - Paa + Pbb))*sb)**2
    #D[dmaac*x1 + dmabc*x2 - dmanc*x3 + dmbbc*x5 + dmbac*x4 - dmbnc*x6,c] // Simplify // FortranForm
    h[2,2] =(1 - 2*na + Paa - Pbb)**2*(-(x1/(2*na + c*(1 - 2*na + Paa - Pbb))**2) -x2/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2 + ((1 - 2*sa)**2*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))**2 - x4/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2 -x5/(2*na + c*(1 - 2*na + Paa - Pbb))**2 - (sb**2*x6)/(-2 + (2 - 2*na + c*(-1 + 2*na - Paa + Pbb))*sb)**2)
    return h

