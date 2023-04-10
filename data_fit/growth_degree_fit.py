import numpy as np
import sys
sys.path.append('..')
import asikainen_model as am
import sympy as sym
from scipy import optimize
from scipy.special import expit


class GrowthFit(object):
    def __init__(self, x_evol, n_evol, na=.5):
        """
        Main function for fitting c, sa, sb from data using the growing model.
        Input:
            x_evol: dictionary of evolution steps. Of the form {t: [[linktype1, tgt_k1,tgt_cnt1, tgt_x1], ...]}, where t is the timestep, linktype one of 'aa', 'ab', 'ba', 'bb', tgt_k the degree of the accepted candidate/target node, and tgt_cnt the number of nodes of degree tgt_k in the group of the target node, and tgt_x the number of times this degree and group were chosen. The list for each t may be of any length.
            n_evol: dictionary of group-level denomitors of the llik function at the corresponding time step. Of the form {t: [Pa_tot, Pb_tot, Ua_tot, Ub_tot]}, where:
                Pa_tot is sum(n_k*k), the sum of degree*(times the degree happens in       group a)
                Pb_tot is sum(n_k*k), the sum of degree*(times the degree happens in goup  b)
                Ua_tot is sum(n_k) the times that each degree happens in group a
                Ub_tot is sum(n_k) the times that each degree happens in group b
    """

        # x_evol and n_evol are the timestamped values for x (new connections by degree and group) and n (group deg distribution)
        self.x_evol = x_evol
        self.n_evol = n_evol
        self.na = na

    def loglik(self, theta): #c, sa, sb, x, n):
        sa, sb, c = theta
        llik = 0
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 1:
                llik -= _lik_step_t_probs(c, sa, sb, nt, xt)
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


    def solve(self, x0=[.5, .5, .5], method='trust-constr'):
        """
        Fit data. Returns array with sa, sb, c
        """
        bounds = optimize.Bounds([.05, 0.05, 0.05], [.95, .95, .95])
        opt = optimize.minimize(self.loglik, x0, method=method, jac=self.grad_loglik, hess=self.hess_loglik, bounds=bounds)
        self.opt = opt
        return opt.x

    def solve_randx0(self, n_x0=10, method='trust-constr'):
        opt = None
        fun = np.inf
        for _ in range(n_x0):
            x0 = np.random.uniform(.1, .9, 3)
            sol = self.solve(x0, method)
            solfun = self.loglik(sol)
            if solfun < fun:
                fun = solfun
                opt = sol
        return opt


    def loglik0(self, theta): #c, sa, sb, x, n):
        sa, sb = theta
        llik = 0
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 10:
                llik -= _lik_step_t_probs(0, sa, sb, nt, xt)
        return llik

    def grad_loglik0(self, theta): #c, sa, sb, x, n):
        grad = np.zeros(3)
        sa, sb = theta
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 10:
                grad -= _gradlik_step_t_probs(0, sa, sb, nt, xt)
        grad = grad[:2]
        return grad

    def hess_loglik0(self, theta): #c, sa, sb, x, n):
        sa, sb = theta
        hess = np.zeros((3, 3))
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 10:
                hess -= _hesslik_step_t_probs(0, sa, sb, nt, xt)
        hess = hess[:2, :2]
        return hess


    def solve_c0(self, x0=[.5, .5], method='trust-constr'):
        bounds = optimize.Bounds([.05, 0.05], [.95, .95])
        opt = optimize.minimize(self.loglik0, x0, method=method, jac=self.grad_loglik0, hess=self.hess_loglik0, bounds=bounds)
        self.opt0 = opt
        return opt.x



def get_dataset(sa, sb, na, c, N, m=10):
    n_iter = 0
    x, n = am.run_growing_varying_m(N, na, c, [sa, sb], ['poisson', m], deg_based=True)
    return x, n

#def get_dataset_varm(sa, sb, na, c, N, m_dist=['poisson', 50], m0=10, em=True):
#    Paa, Pbb, counts = am.run_growing_varying_m(N, na, c, [sa, sb], m_dist, m0=m0, ret_counts=True)
#    if em:
#        obs_counts = np.array(counts)[:, [0, 1, 3, 4]]
#        return Paa, Pbb, obs_counts, counts
#    else:
#        return Paa, Pbb, counts

def _lik_catprobs(Paa, Pbb, na, sa, sb, c):
    nb = 1 - na
    Maa = (c/2*(1+Paa-Pbb) + (1-c)*na)*sa
    Mab = (c/2*(1+Pbb-Paa) + (1-c)*nb)*(1-sa)
    Man = Maa + Mab

    Mba = (c/2*(1+Paa-Pbb) + (1-c)*na)*(1-sb)
    Mbb = (c/2*(1+Pbb-Paa) + (1-c)*nb)*sb
    Mbn = Maa + Mab

    return [Maa, Mab, Man, Mba, Mbb, Mbn]

def _get_h_light(xt, sa, sb):

    try:
        fsrc = xt[0][0]
    except IndexError:
        fsrc = ''


    if fsrc in ['aa', 'ab']:
        return 'a', sa, (1-sa)
    elif fsrc in ['ba', 'bb']:
        return 'b', (1-sb), sb
    else:
        return '', 0, 0


def _lik_step_t_probs(c, sa, sb, nt, xt):
    src, ha, hb = _get_h_light(xt, sa, sb)
    if not src:
        return 0
    #x_i = [linktype, tgtdeg, n_k, x_k(times deg k was selected)]
    Pa_tot = nt[0]
    Pb_tot = nt[1]
    Ua_tot = nt[2]
    Ub_tot = nt[3]

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))
    pi_a = ha * (c*Pa_tot/P_denm + (1-c)*Ua_tot/U_denm)
    pi_b = hb * (c*Pb_tot/P_denm + (1-c)*Ub_tot/U_denm)

    P_tot = pi_a + pi_b
    x_tot = 1 + sum([x[3] for x in xt])

    llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0

    for x in xt:
        tgt_deg = x[1]
        n_k = x[2]
        p_k = c * n_k * tgt_deg / P_denm + (1-c) * n_k / U_denm

        tgt = x[0][1]
        p_k *= ha if tgt == 'a' else hb

        x_k = x[3]
        llik += x_k * np.log(p_k) if p_k > 0 else 0

    return llik

    #Prob of going from src to b
    #for k, xk in xt[src+'b'].items():
    #    nb_k = nt['nb'].get(k, 0)
    #    pib_k = c * nb_k*k/P_denm #if k*P_denm > 0 else 0
    #    pib_k += (1-c)*(nb_k/U_denm) #if U_denm > 0 else 0
    #    llik += xk * (np.log(pib_k) + np.log(hb)) if pib_k > 0 else xk*np.log(hb)


def _gradlik_step_t_probs(c, sa, sb, nt, xt):
    src, ha, hb = _get_h_light(xt, sa, sb)
    if not src:
        return 0
    L = np.zeros(3)

    #x_i = [linktype, tgtdeg, n_k, x_k(times deg k was selected)]
    Pa_tot = nt[0]
    Pb_tot = nt[1]
    Ua_tot = nt[2]
    Ub_tot = nt[3]

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tota = 1 + sum([x[3] for x in xt if x[0][0]=='a'])
    x_totb = 1 + sum([x[3] for x in xt if x[0][0]=='b'])

    #llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0
    Ub = Ub_tot / U_denm
    Ua = Ua_tot / U_denm
    Dca = Pa_tot / P_denm - Ua
    Dcb = Pb_tot / P_denm - Ub
    DUa = Ua - Ub
    DUb = Ub - Ua
    DL_dsa_num = c*(Dca - Dcb) + DUa
    DL_dsa = DL_dsa_num / (sa * DL_dsa_num + c*Dcb + Ub)

    L[0] = 1/sa * sum([x[3] for x in xt if x[0]=='aa']) - (1/(1-sa))*sum([x[3] for x in xt if x[0]=='ab']) - x_tota*DL_dsa
    DL_dsb_num = c*(Dcb - Dca) + DUb
    DL_dsb = DL_dsb_num / (sb * DL_dsb_num + c*Dca + Ua)
    L[1] = 1/sb *sum([x[3] for x in xt if x[0]=='bb']) - (1/(1-sb))*sum([x[3] for x in xt if x[0]=='ba']) - x_totb*DL_dsb

    x_tot = x_tota + x_totb - 1
    if src == 'a':
        num = sa*(Dca - Dcb) + Dcb
        den = c * num + sa*DUa + Ub
    if src == 'b':
        num = sb*(Dcb - Dca) + Dca
        den = c * num + sb*DUb + Ua
    L[2] += -x_tot*num/den

    for x in xt:
        tgt_deg = x[1]
        xk = x[3]
        n_k = x[2]
        p_k = n_k * tgt_deg / P_denm
        u_k = n_k / U_denm

        Dc_k = p_k - u_k
        L[2] += xk * Dc_k / (c*Dc_k + u_k) if c*Dc_k + u_k != 0 else 0

    return L

    #Prob of going from src to b
    #for k, xk in xt[src+'b'].items():
    #    nb_k = nt['nb'].get(k, 0)
    #    pb_k = nb_k*k/P_denm #if k*P_denm > 0 else 0
    #    ub_k = nb_k / U_denm
    #    Dcb_k = pb_k - ub_k
    #    L[2] += xk * Dcb_k / (c*Dcb_k + ub_k) if c*Dcb_k + ub_k > 0 else 0


def _hesslik_step_t_probs(c, sa, sb, nt, xt):
    src, ha, hb = _get_h_light(xt, sa, sb)
    if not src:
        return 0
    H = np.zeros((3, 3))

    Pa_tot = nt[0]
    Pb_tot = nt[1]
    Ua_tot = nt[2]
    Ub_tot = nt[3]

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tota = 1 + sum([x[3] for x in xt if x[0][0]=='a'])
    x_totb = 1 + sum([x[3] for x in xt if x[0][0]=='b'])

    #llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0
    Ub = Ub_tot / U_denm
    Ua = Ua_tot / U_denm
    Dca = Pa_tot / P_denm - Ua
    Dcb = Pb_tot / P_denm - Ub
    DUa = Ua - Ub
    DUb = Ub - Ua

    DL_dsa_num = (c*(Dca - Dcb) + DUa)
    DL_dsa = DL_dsa_num**2 / (sa * DL_dsa_num + c*Dcb + Ub)**2
    H[0, 0] = -1/sa**2 * sum([x[3] for x in xt if x[0]=='aa']) - (1/(1-sa)**2)*sum([x[3] for x in xt if x[0]=='ab']) + x_tota*DL_dsa
    DL_dsb_num = c*(Dcb - Dca) + DUb
    DL_dsb = DL_dsb_num**2 / (sb * DL_dsb_num + c*Dca + Ua)**2
    H[1, 1] = 1/sb * sum([x[3] for x in xt if x[0]=='bb']) - (1/(1-sb))*sum([x[3] for x in xt if x[0]=='ba']) - x_totb*DL_dsb

    H[0, 2] = ((Dca-Dcb)*Ub - Dcb*DUa) / (c*(Dcb +sa*(Dca-Dcb))+sa*DUa+Ub)**2
    H[2, 0] = H[0, 2]

    H[1, 2] = ((Dcb-Dca)*Ua - Dca*DUb) / (c*(Dca +sb*(Dcb-Dca))+sb*DUb+Ua)**2
    H[2, 1] = H[1, 2]

    x_tot = x_tota + x_totb - 1
    if src == 'a':
        num = sa*(Dca - Dcb) + Dcb
        den = c * num + sa*DUa + Ub
    if src == 'b':
        num = sb*(Dcb - Dca) + Dca
        den = c * num + sb*DUb + Ua
    H[2, 2] += x_tot*(num**2)/den**2

    for x in xt:
        tgt_deg = x[1]
        xk = x[3]
        n_k = x[2]

        p_k = n_k * tgt_deg / P_denm
        u_k = n_k / U_denm

        Dc_k = p_k - u_k
        H[2, 2] -= xk * (Dc_k / (c*Dc_k + u_k))**2 if c*Dc_k + u_k != 0 else 0

    #Prob of going from src to b
    #for k, xk in xt[src+'b'].items():
    #    nb_k = nt['nb'].get(k, 0)
    #    pb_k = nb_k*k/P_denm #if k*P_denm > 0 else 0
    #    ub_k = nb_k / U_denm
    #    Dcb_k = pb_k - ub_k
    #    H[2, 2] -= xk * (Dcb_k / (c*Dcb_k + ub_k))**2 if c*Dcb_k + ub_k > 0 else 0

    return H

def c_loglik(sa, sb, x, n):
    vals = [loglik(cv, sa, sb, x, n) for cv in np.linspace(0, 1, 50)]
    return vals

def sa_loglik(c, sb, x, n):
    vals = [loglik(c, sa, sb, x, n) for sa in np.linspace(0, 1, 50)]
    return vals


def _counts_to_coefs(cnt):
    x1, x2, x4, x5 = cnt
    x3 = x1 + x2 + 1
    x6 = x4 + x5 + 1
    coefs = [x1, x2, -x3, x4, x5, -x6]
    return coefs


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

