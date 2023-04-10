import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import plot_policies as pp
import plot_evol as pe
import fit_boards as fb
import rewire_degree_fit as rdf
import pickle
from copy import deepcopy

def get_llik(dt=12):
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    pdata = {}
    for start, end in dates:
        x, n, obs = fb.get_data_in_range(start, end, dt=dt)
        na = obs['na']
        rho_stats(obs)

        RF = rdf.RewireFit(x, n, na)
        opt = RF.solve()
        llik_c = RF.loglik(opt)

        opt0 = RF.solve_c0()
        llik_c0 = RF.loglik_c0(opt0)
        t = 2*(llik_c - llik_c0)
        obs.update({'llik_c': llik_c, 'llik_c0': llik_c0, 'chi': t,
            'sa': opt[0], 'sb': opt[1], 'c': opt[2], 'sa0': opt0[0], 'sb0': opt0[1]})
        print(llik_c, llik_c0)
        print(obs)
        dts = f'{start}-{end}'
        pdata[dts] = obs
    pickle.dump(pdata, open('boards/onion_fit.p', 'wb'))

def rho_stats(obs):
    rho = 2*obs['L'] / obs['N']**2
    rhoa = obs['paa']*rho / obs['na']**2
    rhob = obs['pbb']*rho / (1-obs['na'])**2
    if 'pab' not in obs:
        obs['pab'] = 1-obs['paa'] - obs['pbb']
    rhoab = obs['pab']*rho / (2*obs['na']*(1-obs['na']))
    if rhoa > max([rhob, rhoab]):
        core = 'a'
        cp_meas = (rhoab - rhob) / (rhoab + rhob)
    elif rhob > max([rhoa, rhoab]):
        core = 'b'
        cp_meas = (rhoab - rhoa) / (rhoab + rhoa)
    else:
        core = ''
        cp_meas = np.nan
    obs.update({'rhoa': rhoa, 'rhoab': rhoab, 'rhob': rhob,
        'core': core, 'cp_meas': cp_meas})


def plot_snapshots(dt=12, data=None):
    fig, axs= plt.subplots(1, 3, figsize=(10, 10/3))
    sas, sbs, cs, nas, rhos, taas, tbbs = [], [], [], [], [], [], []
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    if not data:
        for start, end in dates:
            x, n, obs = get_data_in_range(start, end, dt=dt)
            na = obs['na']
            RF = rdf.RewireFit(x, n, .3)
            sa, sb, c = RF.solve()
            print(sa, sb, c)
            sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
            rho = 2*obs['L'] / (obs['N']**2); rhos.append(rho)
            taas.append(obs['paa']), tbbs.append(obs['pbb'])
        data = {'sas': sas, 'sbs': sbs, 'cs': cs, 'nas': nas, 'rhos': rhos, 'taas': taas,  'tbbs': tbbs}
        pickle.dump(data, open('boards/snapshot_data.p', 'wb'))
    else:
        sas, sbs, cs, nas, rhos = data['sas'], data['sbs'], data['cs'], data['nas'], data['rhos']
        taas, tbbs = data['taas'], data['tbbs']
    xvals = ['{}-{}'.format(start+1, end) for start, end in dates]
    axs[0].plot(xvals, sas, '-', color='orangered', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='royalblue', alpha=.5)
    axs[0].plot(xvals, cs, '-', color='darkgreen', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='dimgray', alpha=.5)

    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='orangered')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='royalblue')
    axs[0].plot(xvals, cs, 'o', label=r'$c$', color='darkgreen')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='dimgray')

    pred_ts = [gfp.rewire_fixed_points(c, na, sa, sb) for c, na, sa, sb in zip(cs, nas,    sas, sbs)]
    ptaas = [pred_t[0][0] for pred_t in pred_ts]
    ptbbs = [pred_t[0][1] for pred_t in pred_ts]
    pred_cp = [gfp.rewire_fixed_points_density(c, na, sa, sb, rho*7) for c, na, sa, sb,    rho in zip(cs, nas, sas, sbs, rhos)]
    pred_cpa = [x[1] for x in pred_cp]
    pred_cpb = [x[2] for x in pred_cp]
    obs_cpa = [gfp.cp_correlation_cont(paa, pbb, na, rho*7, 0.01)+np.random.uniform(-.01, .01) for paa, pbb, na, rho in zip(ptaas, ptbbs, nas, rhos)]
    bs_cpb = [gfp.cp_correlation_cont(pbb, paa, 1-na, rho*7, 0.01) for paa, pbb, na, rho  in zip(ptaas, ptbbs, nas, rhos)]

    axs[1].plot(xvals, ptaas, '-', color='orangered', alpha=.5)
    axs[1].plot(xvals, ptbbs, '-', color='royalblue', alpha=.5)
    axs[1].plot(xvals, taas, '-', color='coral', alpha=.5)
    axs[1].plot(xvals, tbbs, '-', color='deepskyblue', alpha=.5)

    axs[1].plot(xvals, ptaas, 'o', label=r'$P^p_{aa}$', color='orangered')
    axs[1].plot(xvals, ptbbs, 'o', label=r'$P^p_{bb}$', color='royalblue')
    axs[1].plot(xvals, taas, 'x', label=r'$P^o_{aa}$', color='coral')
    axs[1].plot(xvals, tbbs, 'x', label=r'$P^o_{bb}$', color='deepskyblue')

    axs[2].plot(xvals, pred_cpa, 'o', label=r'$r^p$', color='palegreen')
    axs[2].plot(xvals, obs_cpa, 'x', label=r'$r^o$', color='slategrey')

    xticks = ['{}-\n{}'.format(s, e) for (s, e) in dates]
    axs[0].set_xticklabels(xticks)
    axs[2].set_xticklabels(xticks)
    axs[1].set_xticklabels(xticks)

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Estimated Parameters')
    axs[0].legend()

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('P-matrix')
    axs[1].set_title('P-matrix')
    axs[1].legend()

    axs[2].set_xlabel('Year')
    axs[2].set_ylabel(r'$r$')
    axs[2].set_title('Core-peripheriness, '+r'$r$')
    axs[2].legend()
    fig.suptitle('Rewiring data fit: Board of Directors')
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('boards/snapshot_abstract_dt{}.pdf'.format(dt))

def plot_snapshots_evolution(dt=12, n_samp=5, data=None):
    fig, axs= plt.subplots(1, 5, figsize=(5*3, 3), sharey=True)
    sas, sbs, cs, nas, rhos, paas, pbss = [], [], [], [], [], [], []
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2011)] #, (2009, 2011)]
    if not data:
        for i, (start, end) in enumerate(dates):
            x, n, obs, net_base, obs_0 = get_data_in_range(start, end, dt=dt, return_net=True)
            na = obs['na']
            r_steps = len(x)
            RF = rdf.RewireFit(x, n, .3)
            sa, sb, c = RF.solve()
            sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
            P0 = (obs_0['paa'], obs_0['pbb'])
            N, L = obs_0['N'], obs_0['L']
            t_size = pe.gfp.rewire_steps(c, na, sa, sb, P0, N, L, 3)
            if t_size is not None:
                t_size = np.mean(t_size)
            else:
                t_size = 2500
            extra = r_steps/(end-start)*.2
            rho = 2*L / (N*(N-1))
            P_obs = (obs['paa'], obs['pbb'])
            rmetal = np.linspace(0, r_steps, end-start+1)
            rmetat = range(start+1, end+2) #in this formulation end includes that year (check)
            rmeta = [(x, y) for x, y in zip(rmetal, rmetat)]
            pe.plot_rewire_predict(sa, sb, c, na, rho, P0=P0, P_obs=P_obs, r_steps=r_steps, ax=axs[i+1], title='{}-{}'.format(start, end+1), extra=extra, rewiring_metadata=rmeta)
            fig.savefig('boards/snapshot_prediction_{}.pdf'.format(dt))
            paas.append(obs['paa']), pbbs.append(obs['pbb'])
        data = {'sas': sas, 'sbs': sbs, 'cs': cs, 'nas': nas, 'rhos': rhos, 'paas': taas,  'pbbs': pbbs}
        pickle.dump(data, open('boards/snapshot_data_4.p', 'wb'))
    else:
        sas, sbs, cs, nas, rhos = data['sas'], data['sbs'], data['cs'], data['nas'], data['rhos']
        paas, pbbs = data['paas'], data['pbbs']
    xvals = ['{}-\n{}'.format(start+1, end+1) for start, end in dates]
     #axs[1].legend()
    axs[0].plot(xvals, sas, '-', color='orangered', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='royalblue', alpha=.5)
     #axs[0].plot(xvals, sas0, '--', color='g', alpha=.5)
     #axs[0].plot(xvals, sbs0, '--', color='b', alpha=.5)
    axs[0].plot(xvals, cs, '-', color='green', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='k', alpha=.5)

    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='orangered')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='royalblue')
    axs[0].plot(xvals, cs, 'o', label=r'$c$', color='green')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='k')

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Boards of Directors in Norway\n Estimated Parameters')
    axs[0].legend()

    fig.tight_layout()
    fig.savefig('boards/snapshot_prediction_{}.pdf'.format(dt))


def plot_policies(dt=12, data=None):
    fig, axs= plt.subplots(1, 5, figsize=(5*3, 3), sharey=True)
    datavars = 'sas,sbs,cs,nas,rhos,pas,pbs,tss,rsteps,pas0,pbs0'
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    if not data:
        data = {k: [] for k in datavars.split(',')}
        for i, (start, end) in enumerate(dates):
            x, n, obs, net_base, obs_0 = fb.get_data_in_range(start, end, dt=dt, return_net=True)
            na = obs['na']
            r_steps = len(x)
            data['rsteps'].append(r_steps)
            RF = fb.rdf.RewireFit(x, n, .3)
            sa, sb, c = RF.solve()
            data['sas'].append(sa); data['sbs'].append(sb); data['cs'].append(c); data['nas'].append(na)
            P0 = (obs_0['paa'], obs_0['pbb'])
            data['pas0'].append(P0[0]); data['pbs0'].append(P0[1])
            P = (obs['paa'], obs['pbb'])
            data['pas'].append(P[0]); data['pbs'].append(P[1])
            N, L = obs_0['N'], obs_0['L']
            rho = 2*L / (N*(N-1))
            data['rhos'].append(rho)
            t_size = pe.gfp.rewire_steps(c, na, sa, sb, P0, N, L, 10)
            if t_size is not None:
                t_size = np.mean(t_size)
            else:
                t_size = 2500
            data['tss'].append(t_size)
            #pe.plot_rewire_predict(sa, sb, c, na, rho, P0=P0, P_obs=P_obs, r_steps=r_steps, ax=axs[i+1], title='{}-{}'.format(start, end+1), extra=extra, rewiring_metadata=rmeta)
            #fig.savefig('boards/snapshot_prediction_{}.pdf'.format(dt))
        #data = {'sas': sas, 'sbs': sbs, 'cs': cs, 'nas': nas, 'rhos': rhos, 'paas': paas,  'pbbs': pbb}
        pickle.dump(data, open('boards/snapshot_data_policies.p', 'wb'))
    sas, sbs, cs, nas, rhos = data['sas'], data['sbs'], data['cs'], data['nas'], data['rhos']
    P0s= (data['pas0'], data['pbs0'])
    Ps = (data['pas'], data['pbs'])
    tss = data['tss']; rsteps = data['rsteps']

    for i, (start, end) in enumerate(dates):
        P0 = (P0s[0][i], P0s[1][i])
        fx_params = {'c': cs[i],
                'sa': sas[i],
                'sb': sbs[i],
                'na': nas[i]}

        #        'P': P0}
        P = (Ps[0][i], Ps[1][i])
        rho = rhos[i]
        #pol_steps = policy_steps(num=4-i, t_size=tss[i], r_steps=rsteps[i], params=params)
        #_ = params.pop('P')
        for j, x in enumerate(['c']): #, 'na', 'sa', 'sb']):

            params = deepcopy(fx_params)
            pol_params = deepcopy(fx_params)
            obs_0 = {'paa': P0[0], 'pbb': P0[1], x: params.pop(x)}
            obs_n = {'paa': P[0], 'pbb': P[1]}

            legend = True if (i == 4) and (j == 0) else False

            pol_steps = get_policies(num=1, t_size=tss[i], r_steps=rsteps[i], params=pol_params, x=x, P0=P0)
            #pp.plot_fullrange_policy_rewire(x, params, rho, obs_0=obs_0, obs_n=obs_n, ax=axs[j][i], nsize=25, pols=pol_steps, legend=legend)
            pp.plot_fullrange_policy_rewire(x, params, rho, obs_0=obs_0, obs_n=obs_n, ax=axs[i], nsize=25, pols=pol_steps, legend=legend)
            fig.savefig('boards/policy_testp.pdf')
    #axs[i].set_title('{}-\n{}'.format(start, end))
    for i, (start, end) in enumerate(dates):
        axs[i].set_title('{}-{}'.format(start, end))
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
   #     for j in range(4):
   #         axs[j][i].spines['top'].set_visible(False)
   #         axs[j][i].spines['right'].set_visible(False)
            #axs[j][i].set_xlabel(r'$x$')
            #axs[j][i].set_ylabel(r'$y$')
    #fig.suptitle('Board of Directors \n Parameter interventions')
    fig.tight_layout()
    fig.savefig('boards/policy_interventions_c.pdf')


def get_policies(num, t_size, r_steps, params, x, P0):
    t = 2 * num * r_steps / t_size
    params['t'] = np.linspace(0, t, max([int(10*t_size), 5000]))
    params['P'] = P0
    xv = params.pop(x)
    xvpols = np.array([xv + i for i in [.1, -.1, .2, -.2, .5, -.5]])
    xvpols = xvpols[(xvpols > 0) & (xvpols < 1)]
    pols = []
    for xvp in xvpols:
        params[x] = xvp
        steps = policy_steps(num, params)
        pol = [xvp, steps]
        pols.append(pol)
    return pols


def policy_steps(num, params):
    ysol = pp.gfp.rewire_path(**params)
    steps = split_last_vals(ysol, num)
    return steps

def split_last_vals(a, n):
    k, m = divmod(len(a), n)
    steps = []
    for i in range(n):
        #st = (a[i*k+min(i, m)], a[(i+1)*k+min(i+1, m)-1])
        st = a[(i+1)*k+min(i+1, m)-1]
        steps.append(st)
    return steps #(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

