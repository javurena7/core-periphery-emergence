import matplotlib.pyplot as plt; plt.ion()
import pandas as pd
import numpy as np
import plot_evol as pe
import plot_policies as pp
import pickle
import fit_airport_db1b as fad
import get_fixed_points as gfp
import rewire_degree_fit as rdf
from copy import deepcopy

def get_data(path='airport/results.txt'):
    """
    Get data and add P/T matrices
    """
    df = pd.read_csv(path, sep='|')
    df['paa'] = df.laa / df.L
    df['pbb'] = df.lbb / df.L
    N = df.N
    df['rho'] = 2*df.L / (N*(N-1))
    df['taa'] = 2*df['paa'] / (1 + df['paa'] - df['pbb'])
    df['tbb'] = 2*df['pbb'] / (1 + df['pbb'] - df['paa'])
    Na = df.N * df.na
    df['rhoa'] = 2*df.laa / Na **2
    df['rhob'] = 2*df.lbb / (N-Na)**2
    df['rhoab'] = df.lab / (Na*(N-Na))
    return df

def get_top(df, toptype='rho', rho_lower=-.05):
    if toptype == 'p':
        df_a = df[(df.paa > 1-df.pbb-df.paa) & (1-df.pbb-df.paa > df.pbb)]
        df_b = df[(df.pbb > 1-df.paa-df.pbb) & (1-df.paa-df.pbb > df.paa)]
    if toptype == 't':
        df_a = df[(df.taa > 1-df.taa) & (1-df.tbb > df.tbb)]
        df_b = df[(df.tbb > 1-df.tbb) & (1-df.taa > df.taa)]
    if toptype == 'rho':
        df['ra'] = (df.rhoab - df.rhob) / (df.rhoab + df.rhob)
        df['rb'] = (df.rhoab - df.rhoa) / (df.rhoab + df.rhoa)
        df_a = df[(df.rhoa > df.rhob) & (df.rhoa > df.rhoab) & (df.ra > rho_lower)]
        df_b = df[(df.rhob > df.rhoa) & (df.rhob > df.rhoab) & (df.rb > rho_lower)]
    else:
        df_a, df_b = None, None
    return df_a, df_b

def likelihood_ratio(a, b, years, dt, opt, opt0):
    """
    For a df with CP, do a likelihood ratio test to see if the model works
    """

    RE = fad.RouteEvol(a, b, years, dt)
    obs0 = RE.get_net_cp_statistics()
    RE.rewire_net()

    RF = rdf.RewireFit(RE.x, RE.n, obs0['na'])
    llik_c = RF.loglik(opt)
    llik_c0 = RF.loglik_c0(opt0)
    t = 2*(llik_c - llik_c0)
    res = {'llik_c': llik_c,
            'llik_c0': llik_c0,
            'chi': t,
            'x': RE.x,
            'n': RE.n}
    return res

def get_likelihood_ratios(df, fpath='airport/onion_llik'):
    """
    For a df of top values, get the likelihood ratio test of not fitting c
    """
    import os
    for i, row in df.iterrows():
        obs = dict(row)
        a, b = _float2list(obs['a']), _float2list(obs['b'])
        pname = ''.join([str(i) for i in a]) + '_' + ''.join([str(i) for i in b]) + '.p'
        fname = f'{fpath}/{pname}'
        if not os.path.exists(fname):
            years = (int(obs['y0']), int(obs['yn']))
            dt = int(obs['dt'])
            opt = (obs['sa'], obs['sb'], obs['c'])
            opt0 = (obs['sa0'], obs['sb0'])
            res = likelihood_ratio(a, b, years, dt, opt, opt0)
            obs.update(res)
            pickle.dump(obs, open(fname, 'wb'))

            chi = res['chi']
            print(f'chi: {chi}')

#### MAIN FUNC
def run_likelihood_ratios(fpath='airport/onion_llik'):
    df = get_data(path='airport/results.txt')
    df_a, _ = get_top(df, 'rho', rho_lower=0)
    df_a = df_a.sort_values('ra', ascending=False)
    #df_b = df_b.sort_values('rb', ascending=False)
    get_likelihood_ratios(df_a, fpath)

def _float2list(f):
    a = str(int(f))
    return list(int(r) for r in a)

def get_cp(df):
    cp_a = []
    cp_b = []
    for i, row in df.iterrows():
        corr_a = gfp.cp_correlation_cont(row.paa, row.pbb, row.na, row.rho, 0)
        corr_b = gfp.cp_correlation_cont(row.pbb, row.paa, 1-row.na, row.rho, 0)
        cp_a.append(corr_a); cp_b.append(corr_b)
    df['cp_a'] = cp_a
    df['cp_b'] = cp_b
    return df


def plot_general(df, suptitle='', name=''):
    fig, axs = plt.subplots(1, 3)

    axs[0].plot(df.na, df.paa, '.')
    axs[1].plot(df.na, df.taa, '.')
    axs[2].plot(df.sa, df.sb, '.')

    axs[0].set_xlabel(r'$n_a$')
    axs[1].set_xlabel(r'$n_a$')
    axs[2].set_xlabel(r'$s_a$')

    axs[0].set_ylabel(r'$P_{aa}$')
    axs[1].set_ylabel(r'$T_{aa}$')
    axs[2].set_ylabel(r'$s_b$')

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig('airport/general_{}.pdf'.format(name))

regions_dict={1: 'New England', 2: 'Mid-Atlantic', 3: 'East North Central',
           4: 'West North Central',
            5: 'South Atlantic',
             6: 'East South Central',
              7: 'West South Central',
               8: 'Mountain',
                9: 'Pacific'}


# 56, 21, 51, 52

def plot_tops(a, b, suptitle='', name=''):
    years = [[1993, 1996], [2000, 2003], [2007, 2010], [2015, 2018]]
    sas, sbs, cs, nas = [], [], [], []
    fig, axs = plt.subplots(1, 5, figsize=(5*3, 4.5))
    for (i, yrs) in enumerate(years):
        sa, sb, c, na, T_obs = fit_ranges(a, b, yrs)
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        pe.plot_rewire_evol(sa, sb, c, na, T_obs=T_obs, ax=axs[i+1], evol_samp=100, t=None, color=None, title='{}-{}'.format(*yrs))
        fig.savefig('airport/{}-{}.pdf'.format(a, b))
        #pe.plot_rewire_evol(sa, sb, 0, na, T_obs=T_obs, ax=axs[1,i+1], evol_samp=100, t=None, color=None, title='No PA')
    plot_
    yrsp = range(len(years))
    axs[0].plot(yrsp, sas, color='orangered')
    axs[0].plot(yrsp, sas, '.', color='orangered', label=r'$s_a$')
    axs[0].plot(yrsp, sbs, color='royalblue')
    axs[0].plot(yrsp, sbs, '.', color='royalblue', label=r'$s_b$')
    axs[0].plot(yrsp, cs, color='green')
    axs[0].plot(yrsp, cs, '.', color='green', label=r'$c$')
    axs[0].plot(yrsp, nas, color='k')
    axs[0].plot(yrsp, nas, '.', color='k', label=r'$n_a$')
    axs[0].set_xlabel('Year Range')
    axs[0].set_ylabel('Params')
    axs[0].legend()
    fig.suptitle('A: {}\nB :{}'.format(regions_dict[int(a)], regions_dict[int(b)]))
    fig.tight_layout()
    fig.savefig('airport/{}-{}.pdf'.format(a, b))

def plot_param_evol(years, sas, sbs, cs, nas, ax):
    yrsp = range(len(years))
    ax.plot(yrsp, sas, color='orangered')
    ax.plot(yrsp, sas, '.', color='orangered', label=r'$s_a$')
    ax.plot(yrsp, sbs, color='royalblue')
    ax.plot(yrsp, sbs, '.', color='royalblue', label=r'$s_b$')
    ax.plot(yrsp, cs, color='green')
    ax.plot(yrsp, cs, '.', color='green', label=r'$c$')
    ax.plot(yrsp, nas, color='k')
    ax.plot(yrsp, nas, '.', color='k', label=r'$n_a$')
    ax.set_xticks(yrsp)
    ax.set_xticklabels('{}-\n{}'.format(yr[0]+1, yr[1]) for yr in years)
    ax.set_xlabel('Year Range')
    ax.set_ylabel('Params')
    ax.legend()


def fit_ranges(a='1', b='2', years=(1993,2002), dt=4):
    ag = [int(i) for i in a.split()]
    bg = [int(i) for i in b.split()]
    sol, obs = fad.full_run(ag, bg, years, outpath='airport/results_{}-{}.txt'.format(a, b), dt=dt)
    sa, sb, c = sol
    na = obs['na'] #* obs['N']
    paa = obs['laa'] / obs['L'] if obs['L'] > 0 else 0
    pbb = obs['lbb'] / obs['L'] if obs['L'] > 0 else 0
    T_obs = fad.p_to_t([[paa, pbb]])[0]
    return sa, sb, c, na, T_obs

# MAIN FUNC 12, 15, 25
def plot_tops_predict_ss(a='1', b='4', suptitle='', name=''):
    """
    Plot four prediction snapshots, using a year of data to predict the network the following year
    """
    years = [[1995, 2000], [2001, 2006], [2007, 2012], [2013, 2018]]
    #years = [[1995, 1997], [2001, 2003], [2007, 2009], [2013, 2015]]
    fig, axs = plt.subplots(1, 5, figsize=(5*3, 4))
    sas, sbs, cs, nas = [], [], [], []
    t_size0 = 2500
    for (i, yrs) in enumerate(years):
        print(yrs)
        sa, sb, c, na, P_obs, P0, r_steps, L, N, rho = fit_ranges_predict(a, b, yrs)
        # r_steps is the number of rewiring steps in the whole period+¿
        print('getting t_size')
        t_size = pe.gfp.rewire_steps(c, na, sa, sb, P0, N, L, 3)
        if t_size is not None:
            t_size = np.mean(t_size)
        else:
            t_size = t_size0
        rmetal = np.linspace(0, r_steps, yrs[1]-yrs[0])
        rmetat = range(yrs[0] + 1, yrs[1] + 1)
        rmeta = [(x, y) for x, y in zip(rmetal, rmetat)]
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        extra = r_steps / len(rmeta) * (.5)
        pe.plot_rewire_predict(sa, sb, c, na, rho, P0=P0, P_obs=P_obs, r_steps=r_steps, ax=axs[i+1], title='{}-{}'.format(yrs[0]+1, yrs[1]), rewiring_metadata=rmeta, extra=extra)
        #axs[i+1].set_xlabel()
        if len(a) < 10:
            fig.savefig('airport/predict_ss_{}-{}.pdf'.format(a, b))
        else:
            fig.savefig('airport/predict_ss_hub.pdf')
        #pe.plot_rewire_evol(sa, sb, 0, na, T_obs=T_obs, ax=axs[1,i+1], evol_samp=100, t=None, color=None, title='No PA'))
    print(years)
    plot_param_evol(years, sas, sbs, cs, nas, ax=axs[0])
    if (len(a) == 1) and (len(b) == 1):
        fig.suptitle('A: {}\nB :{}'.format(regions_dict[int(a)], regions_dict[int(b)]))
    fig.tight_layout()
    if len(a) < 10:
        fig.savefig('airport/predict_ss_{}-{}.pdf'.format(a, b))
    else:
        fig.savefig('airport/predict_ss_hub.pdf')
    return fig, axs



def fit_ranges_predict(a='1', b='2', years=(1993, 1995), dt=4):
    if len(a) < 10:
        ag = [int(i) for i in list(a)] #Comment for hubs
        bg = [int(i) for i in list(b)] #Comment for hubs
    else:
        ag = a; bg = b
    obs0, obs, rew_steps, sol = fad.full_predict(ag, bg, years, dt=dt)
    sa, sb, c = sol
    na = obs['na']
    P_obs = (obs['laa'] / obs['L'], obs['lbb'] / obs['L']) if obs['L'] > 0 else (0, 0)
    P0  = (obs0['laa'] / obs0['L'], obs0['lbb'] / obs0['L']) if obs0['L'] > 0 else (0, 0)
    N = obs['N']
    L = obs['L']
    rho = 2*L / (N*(N-1))
    #T0_obs = fad.p_to_t([[*P0]])[0]
    return sa, sb, c, na, P_obs, P0, rew_steps, L, N, rho

def plot_policies(a='1', b='4', data=None, dt=4):
    """
    Plot four prediction snapshots, using a year of data to predict the network the following year
    """
    years = [[1993, 2000], [2000, 2007], [2007, 2014], [2014, 2021]]
    #years = [[1995, 1997], [2001, 2003], [2007, 2009], [2013, 2015]]
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    t_size0 = 2500
    if not data:
        data = get_policy_data(a, b, years, dt)
    sas, sbs, cs, nas, rhos = data['sas'], data['sbs'], data['cs'], data['nas'],data['rhos']
    P0s= (data['pas0'], data['pbs0'])
    Ps = (data['pas'], data['pbs'])
    tss = data['tss']; rsteps = data['rsteps']

    for i, (start, end) in enumerate(years):
        P0 = (P0s[0][i], P0s[1][i])
        P = (Ps[0][i], Ps[1][i])
        rho = rhos[i]
        fx_params = {'c': cs[i],
                'sa': sas[i],
                'sb': sbs[i],
                'na': nas[i]}
        for j, x in enumerate(['c', 'na', 'sa', 'sb']):
            params = deepcopy(fx_params)
            pol_params = deepcopy(fx_params)
            obs_0 = {'paa': P0[0], 'pbb': P0[1], x: params.pop(x)}
            obs_n = {'paa': P[0], 'pbb': P[1]}
            legend = True if (i == 3) and (j == 0) else False
            pol_steps = pp.get_policies(num=4-i, t_size=tss[i], r_steps=rsteps[i], params=pol_params, x=x, P0=P0)
            pp.plot_fullrange_policy_rewire(x, params, rho, obs_0=obs_0, obs_n=obs_n, ax=axs[j][i], nsize=25, pols=pol_steps, legend=legend)
            fig.savefig('airport/policy_test.pdf'.format(a, b))
    for i, (start, end) in enumerate(years):
        axs[0][i].set_title('{}-\n{}'.format(start, end))
        for j in range(4):
            axs[j][i].spines['top'].set_visible(False)
            axs[j][i].spines['right'].set_visible(False)
    if (len(a) == 1) and (len(b) == 1):
        fig.suptitle('Interventions A: {}\nB :{}'.format(regions_dict[int(a)], regions_dict[int(b)]))
    fig.tight_layout()
    if len(a) < 10:
        fig.savefig('airport/policy_{}-{}_dt{}.pdf'.format(a, b, dt))
    else:
        fig.savefig('airport/policy_hub_dt{}.pdf'.format(dt))

def get_policy_data(a, b, years, dt=4):
    import os
    if len(a) < 10:
        path = 'airport/snapshot_data_policies_a{}_b{}_dt{}.p'.format(a, b, dt)
    else:
        path = 'airport/snapshot_data_policies_hub_dt{}.p'.format(dt)
    if os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
    else:
        datavars = 'sas,sbs,cs,nas,rhos,pas,pbs,tss,rsteps,pas0,pbs0'
        data = {k: [] for k in datavars.split(',')}
        for (i, yrs) in enumerate(years):
            print(yrs)
            sa, sb, c, na, P, P0, r_steps, L, N, rho = fit_ranges_predict(a, b, yrs, dt)
            # r_steps is the number of rewiring steps in the whole period
            data['nas'].append(na); data['rsteps'].append(r_steps); data['sas'].append(sa); data['sbs'].append(sb); data['cs'].append(c); data['rhos'].append(rho)
            data['pas0'].append(P0[0]); data['pbs0'].append(P0[1])
            data['pas'].append(P[0]); data['pbs'].append(P[1])
            print('getting t_size')
            t_size = pe.gfp.rewire_steps(c, na, sa, sb, P0, N, L, 10)
            if t_size is not None:
                t_size = np.mean(t_size)
            else:
                t_size = 2500
            data['tss'].append(t_size)
        pickle.dump(data, open(path, 'wb'))
    return data
