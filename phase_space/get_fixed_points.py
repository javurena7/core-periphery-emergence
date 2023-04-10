import sympy as sm
from scipy.integrate import odeint
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../data_fit/')
import networkx as nx
import simulation_model as am
from pandas import DataFrame
from itertools import product

def rewire_fixed_points(c, na, sa, sb):
    """
    Get fixed points for the rewiring model in terms of the P-matrix given four parameter
    All parameters are values between 0 and 1
    Input:
        c: amount of preferential attachment
        na: relative size of group a
        sa: assortative attachment of group a
        sb: assortative attachment of group b
    Output:
        equilibria: list of fix points in terms of P-matrices (either 1 or 3 fixed points)
    """
    paa, pbb = sm.symbols('paa, pbb', negative=False)

    maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
    mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
    mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
    mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

    taa = 2*paa / (paa + 1 - pbb)
    tbb =  2*pbb / (pbb + 1 - paa)

    Paa = na*maa - na*taa*(maa + mab)
    Pbb = (1-na)*mbb - (1-na)*tbb*(mba + mbb)

    PaaEqual = sm.Eq(Paa, 0)
    PbbEqual = sm.Eq(Pbb, 0)

    equilibria = sm.solve((PaaEqual, PbbEqual), paa, pbb)
    equilibria = check_solutions(equilibria)
    #equilibria = p_to_t(equilibria)

    return equilibria

def growth_fixed_points(c, na, sa, sb):
    """
    Get fixed points for the growing model in terms of the P-matrix given four parameter
    All parameters are values between 0 and 1
    Input:
        c: amount of preferential attachment
        na: relative size of group a
        sa: assortative attachment of group a
        sb: assortative attachment of group b
    Output:
        equilibria: list of fix points in terms of P-matrices (either 1 or 3 fixed points)
    """
    paa, pbb = sm.symbols('paa, pbb', negative=False)
    maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
    mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
    mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
    mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

    DLaa = na*maa
    DLbb = (1-na)*mbb
    DL = na*(maa+mab) + (1-na)*(mba+mbb)

    PaaEqual = sm.Eq(DLaa/DL, paa)
    PbbEqual = sm.Eq(DLbb/DL, pbb)
    equilibria = sm.solve((PaaEqual, PbbEqual), paa, pbb) #This might yield a dict
    if isinstance(equilibria, dict):
        equilibria = [(equilibria.get(paa, np.nan), equilibria.get(pbb, np.nan))]
    equilibria = check_solutions(equilibria)
    return equilibria


#MAIN FUNC
def rewire_path(c, na, sa, sb, P, t):
    """
    Path of p-matrices for the rewiring model:
    Input:
        c, na, sa, sb: rewiring model paramters
        P: (tuple or list with [paa, pbb] values) initial conditions
        t: list or array of time values
    Output:
        ysol: list of [paa, pbb] solutions for the specified t values
    """
    paa, pbb = P #p_from_G(G0, Na)
    y0 = np.array([paa, pbb])

    def rewire_model(y, t):
        paa, pbb = y
        maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
        mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
        mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
        mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

        taa = 2*paa / (paa + 1 - pbb)
        tbb =  2*pbb / (pbb + 1 - paa)

        dPaa = na*maa - na*taa*(maa + mab)
        dPbb = (1-na)*mbb - (1-na)*tbb*(mba + mbb)

        return np.array([dPaa, dPbb])
    ysol = odeint(rewire_model, y0, t)
    return ysol

def rewire_fixed_points_density(c, na, sa, sb, rho=.1):
    """
    Get fixed points for the rewiring model in terms of local densities,
    and evaluate Borgatti and Everett's correlation 'r'.
    Input:
        c: amount of preferential attachment
        na: relative size of group a
        sa: assortative attachment of group a
        sb: assortative attachment of group b
        rho: network density
    Output:
        densities: list of fix points in terms of local densities (either 1 or 3 fixed points)
        corrs_a: 'r' for all fixed points when ideal a is a core
        corrs_b: 'r' for all fixed points when ideal b is a core
    """
    paa, pbb = sm.symbols('paa, pbb', negative=False)

    maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
    mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
    mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
    mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

    taa = 2*paa / (paa + 1 - pbb)
    tbb =  2*pbb / (pbb + 1 - paa)

    Paa = na*maa - na*taa*(maa + mab)
    Pbb = (1-na)*mbb - (1-na)*tbb*(mba + mbb)

    PaaEqual = sm.Eq(Paa, 0)
    PbbEqual = sm.Eq(Pbb, 0)

    equilibria = sm.solve((PaaEqual, PbbEqual), paa, pbb)
    equilibria = check_solutions(equilibria)
    corrs_a = [rmax(paa, pbb, na, rho) for (paa, pbb) in equilibria]
    corrs_b = [rmax(pbb, paa, 1-na, rho) for (paa, pbb) in equilibria]
    densities = p_to_rho(equilibria, rho, na)
    return densities, corrs_a, corrs_b


def growth_fixed_points_density(c, na, sa, sb, rho=.1):
    """
    Get fixed points for the growing model in terms of local densities,
    and evaluate Borgatti and Everett's correlation 'r'.
    Input:
        c: amount of preferential attachment
        na: relative size of group a
        sa: assortative attachment of group a
        sb: assortative attachment of group b
        rho: network density
    Output:
        densities: list of fix points in terms of local densities (either 1 or 3 fixed points)
        corrs_a: 'r' for all fixed points when ideal a is a core
        corrs_b: 'r' for all fixed points when ideal b is a core
    """
    equilibria = growth_fixed_points(c, na, sa, sb)
    corrs_a = [rmax(paa, pbb, na, rho) for (paa, pbb) in equilibria]
    corrs_b = [rmax(pbb, paa, 1-na, rho) for (paa, pbb) in equilibria]
    densities = p_to_rho(equilibria, rho, na)
    return densities, corrs_a, corrs_b

def cp_correlation(laa, lab, lbb, n, na, a=.01):
    """
    Borgatti and Everett's discrete correlation to ideal matrix, based on number
    of links per group (laa, lab, lbb), total number of nodes (n), total number
    of nodes in group a (na) and alpha value (0 < a < 1, usually low)
    """
    nb = n - na
    N = n*(n-1)
    exy = 2*laa / N + 2*lab*a / N
    ex = 2*(laa + lab + lbb) / N
    ey = na*(na-1) / N + (a*2*nb*na) / N
    sx = ex*(1-ex)
    sy = ey*(1-ey)
    corr = (exy - ex*ey) / np.sqrt(sx*sy)
    return corr

def cp_correlation_cont(paa, pbb, na, rho, a):
    """
    Borgatti and Everett's continious correlation to ideal matrix, based on P matrix (paa, pbb), minority size, density and alpha (0<a<1, usually low)
    Note: rho is a density
    of nodes in group a (na) and alpha value (0 < a < 1, usually low)
    Note:
    For the correlation value to be valid, they must satisfy
    paa <= na**2/rho, pbb <= (1-na)**2/rho
    """
    # rho is the newtork density rho=2L/N(N-1)
    pab = 1 - paa - pbb
    ex = rho
    ey = na**2 + 2*a*na*(1-na)
    exy = (paa + a*pab)*rho
    sx = ex*(1-ex)
    sy = ey*(1-ey)
    corr = (exy - ex*ey) / np.sqrt(sx*sy)
    return corr

def rmax(paa, pbb, na, rho):
    """
    Borgatti and Everett's continious correlation to ideal matrix,
    maximizing over alpha values.
    Where,
    paa: fraction of links in group a
    pbb: fraction of links in group b
    na: minority fraction
    rho: network density
    Note: for the correlation value to be valid, it must satisfy
    paa <= |na**2/rho and pbb <= (1-na)**2/rho
    """
    vals = [(cp_correlation_cont(paa, pbb, na, rho, a), a) for a in np.linspace(0, 1, 1000)]
    # For some rho, alpha values the resulting networks are impossible
    # Since this is a result of the choice of rho and not of the model, we cut off those values (see supplementary material)
    vals = [(r, a) for r, a in vals if (r <= 1) and (r >= -1)]
    if vals:
        return max(vals, key=lambda x: x[0])
    else:
        return (np.nan, np.nan)


def p_to_rho(sols, rho, na):
    """
    Convert list of solutiosns in terms of P-matrices, into local densities
    Needs a set network density and minoirty size
    """
    rhos = []
    for sol in sols:
        paa, pbb = sol
        rho_aa = rho * paa / na**2
        rho_bb = rho * pbb / (1-na)**2
        rho_ab = rho * (1-paa-pbb) / (na*(1-na))
        rhos.append((rho_aa, rho_ab, rho_bb))
    return rhos


def rewire_simul_simple(c, na, sa, sb, P, t, N, L, return_net=False):
    """
    Simulate t time steps starting from initial condition P. Uses model paramters (c, na, sa, sb), simulated network specifications (N, L)
    """
    paa, pbb = P
    Na = int(na*N)
    Nb = int((1-na)*N)
    N = Na + Nb
    Laa = L*paa
    Lbb = L*pbb
    Lab = L - Laa - Lbb
    p_sbm = [[2*Laa/(Na*(Na-1)), Lab/(Na*Nb)], [Lab/(Na*Nb), 2*Lbb/(Nb*(Nb-1))]]
    sizes = [Na, Nb]
    G = nx.stochastic_block_model(sizes=sizes, p=p_sbm)
    p_simul = p_from_G(G, Na)
    print(f'p orig: {P}')
    i = 0
    while i < t:
        n_link, d_link, _, _, _ = am._rewire_candidates_exact(G, N, Na, c, [sa, sb])
        if d_link and n_link:
            G.add_edge(*n_link)
            G.remove_edge(*d_link)
        i += 1
    if return_net:
        return G
    else:
        P = p_from_G(G, Na)
        return P

def rewire_simul_save_steps(c, na, sa, sb, P, t, N, L, return_net=False):
    """
    t is the expected number of times all links are chosen
    """
    T = t * N
    paa, pbb = P
    Na = int(na*N)
    Nb = int((1-na)*N)
    N = Na + Nb
    Laa = L*paa
    Lbb = L*pbb
    Lab = L - Laa - Lbb
    p_sbm = [[2*Laa/(Na*(Na-1)), Lab/(Na*Nb)], [Lab/(Na*Nb), 2*Lbb/(Nb*(Nb-1))]]
    sizes = [Na, Nb]
    G = nx.stochastic_block_model(sizes=sizes, p=p_sbm)
    p_simul = p_from_G(G, Na)
    Ps = []
    print(f'p orig: {P}')
    print(f'p simul: {p_simul}')
    i = 0
    while i < T:
        n_link, d_link, _, _, _ = am._rewire_candidates_exact(G, N, Na, c, [sa, sb])
        if d_link and n_link:
            G.add_edge(*n_link)
            G.remove_edge(*d_link)
        if i % N == 0:
            P = p_from_G(G, Na)
            Ps.append(P)
        i += 1
    return np.array(Ps)

def rewire_steps(c, na, sa, sb, P0, N, L, n_steps=10, t_size=5000):
    """
    Returns the number of simulated rewiring steps in a time step (in terms of mean-field equations),
    Note: in genearl, there are around 2500 rewiring steps per MFE timestep
    """
    t = np.linspace(0, n_steps, t_size)
    path = rewire_path(c, na, sa, sb, P0, t)
    rsteps = list(range(0, t_size, int(t_size/n_steps))) + [-1]
    P_steps = [path[i, :] for i in rsteps]
    try:
        n_ests = [1] + rewire_until_p(c, na, sa, sb, P0, N, L, P_steps)
    except:
        n_ests = [None]
    if np.all(n_ests):
        n_ests = [i-j for i, j in zip(n_ests[1:], n_ests[:-1])]
        return n_ests
    else:
        return None


def rewire_until_p(c, na, sa, sb, P0, N, L, P_steps):
    """
    Tool validation between simulations and mean-field equations
    Given a sequence of P-matrices (from, e.g., rewire_path),
    find the number of steps needed to reach each p-matrix
    Note: since simulations are stochastic, this likely requires manual adjustment
    in terms of combinations of P-steps, P0 and distance

    c, na, sa, sb: model parameters
    P0: initial conditions for p-matrix (tuple, array or list [paa, pbb])
    N, L: number of nodes and links for simulations
    P_steps: sequence of P-matrices that the simulations should follow
        e.g., [(paa_0, pbb_0), (paa_1, pbb_1), ...]
    """
    paa, pbb = P0
    Na, Nb = int(na*N), int((1-na)*N)
    N = Na + Nb
    Laa, Lbb = L*paa, L*pbb
    Lab = L - Laa - Lbb
    p_sbm = [[2*Laa/(Na*(Na-1)), Lab/(Na*Nb)], [Lab/(Na*Nb), 2*Lbb/(Nb*(Nb-1))]]
    sizes = [Na, Nb]
    G = nx.stochastic_block_model(sizes=sizes, p=p_sbm)
    p_simul = p_from_G(G, Na)
    i, j = 0, 1
    steps = []
    while j < len(P_steps):
        n_link, d_link, _, _, _ = am._rewire_candidates_exact(G, N, Na, c, [sa, sb])
        if d_link and n_link:
            i += 1
            G.add_edge(*n_link)
            G.remove_edge(*d_link)
            Laa, Lab, Lbb = link_counts(n_link, Na, Laa, Lab, Lbb) #new link
            Laa, Lab, Lbb = link_counts(d_link, Na, Laa, Lab, Lbb, False) #deleted link
            Paa = Laa / L
            Pbb = Lbb / L
            dist = np.sqrt((Paa - P_steps[j][0])**2 + (Pbb - P_steps[j][1])**2)
            if dist < 10e-2:
                steps.append(i)
                j += 1
            elif i > j*10000: #Safguard: each matrix from P-steps is at most 10000 steps away
                print('Real: {}-{}| Sim: {}-{}'.format(P_steps[j][0], P_steps[j][1], Paa, Pbb))
                steps.append(None)
                j += 1
    return steps

def link_counts(link, Na, Laa, Lab, Lbb, add=True):
    """
    Changes in group mixing when a link is added (add=True) or deleted (False)
    """
    src = 'a' if link[0] <= Na else 'b'
    tgt = 'a' if link[1] <= Na else 'b'
    if src == tgt:
        if src == 'a':
            Laa = Laa + 1 if add else Laa - 1
        else:
            Lbb = Lbb + 1 if add else Lbb - 1
    else:
        Lab = Lab + 1 if add else Lab - 1

    return Laa, Lab, Lbb


def growth_steps(c, na, sa, sb, P0, L0, N, n_steps=1000, t_size=10000):
    """
    Returns the number of evolution steps in a time step (usually around 1/2 and evolution step)
    """
    t = np.linspace(0, n_steps, 10000)
    path = growth_path(c, na, sa, sb, P0, L0, t)
    rsteps = list(range(0, 10000, int(10000/n_steps)))
    P_steps = [path[i, 2] for i in rsteps]
    #n_ests, added_seq = grow_until_p(c, na, sa, sb, P0, L0, N, P_steps, m_avg=10)
    n_ests = [i-j for i, j in zip(P_steps[1:], P_steps[:-1])]
    return n_ests


def grow_until_p(c, na, sa, sb, P0, L, N, P_steps, m_avg):
    paa, pbb = P0
    Na, Nb = int(na*N), int((1-na)*N)
    N = Na + Nb
    Laa, Lbb = L*paa, L*pbb
    Lab = L - Laa - Lbb
    G, Na, dist = am.ba_starter(N, na, sa, sb)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, 5))
    for tgt in target_list:
        _ = sources.remove(tgt)

    i, j = 0, 1
    steps = []
    added_seq = []
    m_add = 2*m_avg
    while i < len(sources):
        try:
            counts = am.grow_ba_two(G, sources, target_list, dist, m_avg, c, ret_counts=True, n_i={}, Na=0)
        except:
            import pdb; pdb.set_trace()
        Laa, Lab, Lbb, leftover = add_growth_counts(counts, Laa, Lab, Lbb)
        added = sum(counts) - leftover
        added_seq.append(added)
        i += 1
        L = Laa + Lab + Lbb
        Paa = Laa / L
        Pbb = Lbb / L
        cp_dist = np.sqrt((Paa - P_steps[j][0])**2 + (Pbb - P_steps[j][1])**2)
        if cp_dist < 10e-3:
            steps.append(i)
            j += 1
        elif i > j*3000:
            print('Real: {}-{}| Sim: {}-{}'.format(P_steps[j][0], P_steps[j][1], Paa, Pbb))
            steps.append(None)
            j += 1
    return steps, added_seq


def add_growth_counts(counts, Laa, Lab, Lbb):
    #Classify counts if the added links are: AA, AB, AN, BA, BB, BN
    Laa += counts[0]
    Lab += counts[1] + counts[3]
    Lbb += counts[4]
    leftover = counts[2] + counts[5]
    return Laa, Lab, Lbb, leftover


def rewire_simul_n(c, na, sa, sb, P, t, N, L, n_samp):
    """
    Obtain n_samp samples of size t when P is the initial condition
    """
    p = []
    for i in range(n_samp):
        print(f'Getting simul {i} / {n_samp}')
        psamp = rewire_simul_simple(c, na, sa, sb, P, t, N, L)
        p.append(psamp)
    return p

def grow_simul_simple(c, na, sa, sb, N, m):
    """
    Growing model. For each of N nodes:
        (1) Select m candidate nodes in the netwod (target_list)
            a. With probability c, select via preferntial attachment
                and w.p. 1-c, uniformly random
            b. Accept the candidates with assortative attachment parameter
        (2) If the nodes are accepted, add links
    Input:
        c: probability of preferential attachment (0, 1)
        na: minority fraction (0, 1)
        sa: assorative attachment for group a [0<sa<.5 disassortative; .5<sa<1 assorative]
        sb: assorative attachment for group b [0<sb<.5 disassortative; .5<sb<1 assorative]
    Output:
        P: group mixing matrix diagonal: [Paa, Pbb], where Paa is the fraction of
            links in group a
    """
    G, Na, dist = am.ba_starter(N, na, sa, sb)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, 5))
    while len(sources) > 0:
        am.grow_ba_two(G, sources, target_list, dist, m, c, ret_counts=False, n_i={}, Na=0)

    P = p_from_G(G, Na)
    return P

def grow_simul_n(c, na, sa, sb, N, m, n_samp):
    p = []
    for i in range(n_samp):
        print(f'Getting simul {i} / {n_samp}')
        psamp = grow_simul_simple(c, na, sa, sb, N, m)
        p.append(psamp)
    return p


def p_from_G(G, Na):
    """
    Get group mixing matrix (P) from a network G with Na nodes in group a
    Output:
        paa, pbb: group mixing matrix; fraction of links in groups a and b.
    """
    paa, pbb, pab = 0, 0, 0
    for edge in G.edges():
        if edge[0] >= Na and edge[1] >= Na:
            pbb += 1
        elif edge[0] < Na and edge[1] < Na:
            paa += 1
        else:
            pab += 1
    tot = paa + pab + pbb
    paa /= tot if tot > 0 else 1
    pbb /= tot if tot > 0 else 1
    return paa, pbb


def check_solutions(equilibria):
    """
    Check that solutions are real and in the [0, 1] range
    """
    eqs = []
    for sol in equilibria:
        s1 = sol[0]
        s2 = sol[1]
        if s1.is_real and s2.is_real:
            if valid_sol(s1, s2):
                eqs.append((s1, s2))
        elif sm.im(s1) < 1.e-8 and sm.im(s2) < 1.e-8:
            s1 = sm.re(s1); s2 = sm.re(s2)
            if valid_sol(s1, s2):
                eqs.append((s1, s2))
    return eqs

def valid_sol(s1, s2):
    s1_val = number_in_range(s1)
    s2_val = number_in_range(s2)
    s_val = number_in_range(s1 + s2)
    if s1_val and s2_val and s_val:
        return True
    else:
        return False

def number_in_range(num):
    if (num >= 0) and (num <= 1):
        return True
    else:
        return False


def p_to_t(equilibria):
    """
    Convert p matrix from equilibria into T-matrix
    """
    ts = []
    for ps in equilibria:
        pa, pb = ps[0], ps[1]
        taa = 2*pa / (pa + 1 - pb)
        tbb = 2*pb / (pb + 1 - pa)
        ts.append((taa, tbb))
    return ts


def growth_path(c, na, sa, sb, P, L0, t):
    paa, pbb = P #p_from_G(G0, Na)
    y0 = np.array([paa, pbb, L0])

    def growth_model(y, t):
        paa, pbb, L = y
        maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
        mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
        mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
        mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

        dlaa = na*maa
        dlbb = (1-na)*mbb
        dL = na*(maa + mab) + (1-na)*(mba + mbb)

        dPaa = (dlaa - dL*paa) / L
        dPbb = (dlbb - dL*pbb) / L

        return np.array([dPaa, dPbb, dL])

    ysol = odeint(growth_model, y0, t)
    return ysol


def plot_test_cont(adjust=False, a=0):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    nas = (.1, .2, .3, .4, .5)
    rhos = (.001, .005, .01, .05, .1)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3), sharex=True, sharey=True)
    na_corr = {}
    for i, na in enumerate(nas):
        for j, rho in enumerate(rhos):
            vals = np.zeros((101, 101))
            vals[:] = np.nan
            for paa, pbb in product(ps, ps):
                if paa + pbb <= 100:
                    if not adjust:
                        corr = cp_correlation_cont(paa/100, pbb/100, na, rho, a)
                    else:
                        corr = cp_correlation_cont_correctdens(paa/100, pbb/100, na, rho)
                    corr = corr if corr <= 1 else np.nan
                    vals[paa, pbb] = corr
                    #vals[pbb, paa] = np.nan
            sns.heatmap(vals, ax=axs[j, i], center=0, vmin=-.2, vmax=1)
            axs[j,i].set_xlabel(r'$P_{bb}$')
            axs[j,i].set_ylabel(r'$P_{aa}$')
            axs[j,i].set_title(r'$n_a=$' + f'{na}\n' + r'$\rho=$' + f'{rho}')
            na_corr[str(na)] = vals
    xticks = axs[j,i].get_xticks()
    axs[j,i].invert_yaxis()
    yticks = axs[j,i].get_yticks()
    axs[j,i].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
    axs[j,i].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
    if not adjust:
        fig.suptitle('Continuous correlation to ideal CP matrix')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_mat_a{}.pdf'.format(a))
    else:
        fig.suptitle('Continuous adjusted correlation to ideal CP matrix')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_mat_adjust.pdf')

    return na_corr, (fig, axs)

def plot_test_discrete():
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    nas = (.1, .2, .3, .4, .5)
    rhos = (.001, .005, .01, .05, .1)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3), sharex=True, sharey=True)
    na_corr = {}
    for i, na in enumerate(nas):
        vals = np.zeros((101, 101))
        vals[:] = np.nan
        for paa, pbb in product(ps, ps):
            if paa + pbb <= 100:
                corr = cp_correlation_cont(paa/100, pbb/100, na, 0)
                vals[paa, pbb] = corr
                #vals[pbb, paa] = np.nan
        #vals = DataFrame(vals,columns=['paa', 'pbb', 'cr'])
        sns.heatmap(vals, ax=axs[i], center=0)
        axs[i].set_xlabel(r'$P_{bb}$')
        axs[i].set_ylabel(r'$P_{aa}$')
        axs[i].invert_yaxis()
        axs[i].set_title(r'$n_a=$' + f'{na}' )
        na_corr[str(na)] = vals
    xticks = axs[i].get_xticks()
    yticks = axs[i].get_yticks()
    axs[i].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
    axs[i].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
    fig.suptitle('Continuous correlation to ideal CP matrix')
    fig.tight_layout()
    fig.savefig('cp_measure_plots/disc_corr_cp_mat.pdf')

def plot_test_cont_rho_na(adjust=False):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    paas = (1, .75, .5, .25, .1)
    pbbs = (0, .25, .5, .75, .9)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3))
    na_corr = {}
    for i, paa in enumerate(paas):
        for j, pbb in enumerate(pbbs):
            vals = np.zeros((101, 101))
            vals[:] = np.nan
            if paa + pbb <= 1:
                for na, rho in product(ps, ps):
                    if not adjust:
                        corr = cp_correlation_cont(paa, pbb, na/100, rho/100, 0)
                    else:
                        corr = cp_correlation_cont_correctdens(paa, pbb, na/100, rho/100)
                    corr = corr if np.abs(corr) <= 1 else np.nan
                    vals[na, rho] = corr
                    #vals[pbb, paa] = np.nan
                sns.heatmap(vals, ax=axs[i, j], center=0, vmin=-1, vmax=1)
                axs[i,j].set_xlabel(r'$\rho$')
                axs[i,j].set_ylabel(r'$n_{a}$')
                axs[i,j].set_title(r'$P_{aa}=$' + f'{paa}    ' + r'$P_{bb}=$' + f'{pbb}')
                xticks = axs[i,j].get_xticks()
                axs[i,j].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
                axs[i,j].invert_yaxis()
                yticks = axs[i,j].get_yticks()
                axs[i,j].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
            else:
                axs[i,j].axis('off')
    fig.tight_layout()
    if not adjust:
        fig.suptitle('Continuous correlation to ideal CP matrix\n varying ' +r'$n_a$' ' and '+r'$\rho$')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_na_rho.pdf')
    else:
        fig.suptitle('Continuous adjusted correlation to ideal CP matrix\n varying ' +r'$n_a$' ' and '+r'$\rho$')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_na_rho_adjust.pdf')

def plot_test_cont_local_dens(adjust=False, a=0):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    rho_bbs = (0, .05, .1, .25, .5)
    rho_abs = (0, .05, .1, .25, .5)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3))
    na_corr = {}
    for i, rho_bb in enumerate(rho_bbs):
        for j, rho_ab in enumerate(rho_abs):
            vals = np.zeros((101, 101))
            vals[:] = np.nan
            for Na, rho_aa in product(ps, ps):
                na = Na / 100
                rho = (rho_aa/100)*na**2 + 2*rho_ab*na*(1-na) + rho_bb*(1-na)**2
                paa = (rho_aa/100)*na**2 / rho if rho > 0 else 0
                pbb = rho_bb*(1-na)**2 / rho if rho > 0 else 0
                if not adjust:
                    corr = cp_correlation_cont(paa, pbb, na, rho, a)
                else:
                    corr = cp_correlation_cont_correctdens(paa, pbb, na, rho)
                corr = corr if np.abs(corr) <= 1 else np.nan
                vals[Na, rho_aa] = corr
                #vals[pbb, paa] = np.nan
            sns.heatmap(vals, ax=axs[i, j], center=0, vmin=-1, vmax=1)
            axs[i,j].set_xlabel(r'$\rho_{aa}$')
            axs[i,j].set_ylabel(r'$n_{a}$')
            axs[i,j].set_title(r'$\rho_{ab}=$' + f'{rho_ab}    ' + r'$\rho_{bb}=$' + f'{rho_bb}')
            xticks = axs[i,j].get_xticks()
            axs[i,j].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
            axs[i,j].invert_yaxis()
            yticks = axs[i,j].get_yticks()
            axs[i,j].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
    fig.tight_layout()
    if not adjust:
        fig.suptitle('Continuous correlation to ideal CP matrix\n varying local densitites')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_local_dens_a{}.pdf'.format(a))
    else:
        fig.suptitle('Continuous adjusted correlation to ideal CP matrix\n varying local densitites')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_local_dens_adjust.pdf')


