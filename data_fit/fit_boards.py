import numpy as np
import networkx as nx
import rewire_degree_fit as rdf
import growth_degree_fit as gdf
import matplotlib.pyplot as plt; plt.ion()
from collections import Counter
import sys; sys.path.append('..')
import get_fixed_points as gfp
import plot_evol as pe
path = '../../data/board/'

"""
In December 2003, the Norwegian Government included in the Public Limited Companies Act that the boards of directors of companies bound by the law should be gender balanced. In a dialog between the Norwegian Government and the private sector, it was agreed that the amendment should be withdrawn if the companies voluntarily complied by July 2005. However, the proportion of women had only risen to 16% (The Norwegian Government, 2008).
Therefore, the Norwegian Government introduced in January 2006 a gender representation law requiring public limited companies to compose their boards of directors with at least 40% of each sex within a two-year period.
"""

def read_net(npath):
    net = nx.read_edgelist(npath)
    return net

def parse_groups(path=path+'people_list.csv'):
    groups = {}
    with open(path, 'r') as r:
        line = r.readline().strip()
        while line:
            node, group = line.strip().split()
            groups[node] = group
            line = r.readline()
    return groups

def edge_type(edge, groups, G0, G1):
    edge = edge_reorder(edge, G0, G1)

    g1 = groups[edge[0]]
    g2 = groups[edge[1]]
    a1 = 'a' if g1 == '2' else 'b'
    a2 = 'a' if g2 == '2' else 'b'

    if a1 == a2:
        if a1 == 'b':
            return 'bb'
        else:
            return 'aa'
    else:
        if np.random.rand() < .5: #G.degree(edge[0]) <= G.degree(edge[1]):
            return a1 + a2
        else:
            return a2 + a1

def edge_reorder(edge, G0, G1):
    deg_0 = G0.degree(edge[0])
    deg_1 = G0.degree(edge[1])

    if isinstance(deg_0, int):
        if isinstance(deg_1, int):
            if deg_0 < deg_1:
                return edge
            elif deg_1 < deg_0:
                return (edge[1], edge[0])
            else:
                return edge_rand_perm(edge)
        else:
            return (edge[1], edge[0])
    elif isinstance(deg_1, int):
        return edge
    else:
        #print('two new nodes')
        return edge_rand_perm(edge)


def edge_rand_perm(edge):
    if np.random.rand() < .5:
        return edge
    else:
        return (edge[1], edge[0])



def net_diff(net, nnet, groups):
    x_i = []
    noadd = 0
    degs = Counter([v for k, v in net.degree()])
    for edge in nnet.edges():
        if not net.has_edge(*edge):
            x = []
            et = edge_type(edge, groups, net, nnet)

            tgt_deg = net.degree(edge[1])
            if isinstance(tgt_deg, int):
                x.append(et)
                x.append(tgt_deg)
                x.append(degs[tgt_deg])
                x_i.append(x)
            else:
                noadd += 1
    print('     +{}'.format(len(x_i)))
    print('     -{}'.format(noadd))
    return x_i

def net_tots(net, groups):
    n_i = {'na': {}, 'nb': {}}
    for node in net:
        deg = net.degree(node)
        if groups[node] == '2':
            n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
        else:
            n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
    pa = sum([k*v for k, v in n_i['na'].items()])
    pb = sum([k*v for k, v in n_i['nb'].items()])
    ua = sum(n_i['na'].values())
    ub = sum(n_i['nb'].values())
    n = [pa, pb, ua, ub]
    return n

def parse_nets(net0, net1, x, n, groups):
    n_ = net_tots(net0, groups)
    x_i = net_diff(net0, net1, groups)
    n_i = [n_ for _ in range(len(x_i))]

    x += x_i
    n += n_i

def update_net0(net0, net1, time, dt):

    for (n1, n2) in net1.edges():
        net0.add_edge(n1, n2, t=time)

    for (n1, n2) in net0.edges():
        t = net0[n1][n2].get('t', 0)
        if t < time - dt:
            net0.remove_edge(n1, n2)

    return net0
#TODO: get_last_obs(net0)

def update_na(na, net0, groups):
    for node in net0.nodes():
        g = groups[node]
        if g == '2':
            na['a'].add(node)
        else:
            na['b'].add(node)

def get_last_obs(net0, groups):
    """
    Get Taa, Tbb, Paa, Pbb
    """
    paa, pbb, pab = 0, 0, 0
    for edge in net0.edges():
        g1 = groups[edge[0]]
        g2 = groups[edge[1]]
        a1 = 'a' if g1 == '2' else 'b'
        a2 = 'a' if g2 == '2' else 'b'

        if a1 == a2:
            if a1 == 'a':
                paa += 1
            else:
                pbb += 1
        else:
            pab += 1
    tot = paa + pab + pbb
    paa /= tot
    pbb /= tot
    pab /= tot

    taa = 2*paa / (2*paa + pab)
    tbb = 2*pbb / (2*pbb + pab)

    obs = {'paa': paa,
            'pab': pab,
            'pbb': pbb,
            'taa': taa,
            'tbb': tbb}

    return obs


def get_data_in_range(start=2002, end=2007, dt=12, return_net=False, return_times=False):
    net0 = None
    groups = parse_groups()
    x, n, na = [], [], {'a': set(), 'b': set()}
    time = 0
    timings = {}
    for year in range(start, end + 1):
        for month in range(1, 13):
            if (year > 2002 and year < 2011) or (year == 2002 and month > 4) or (year == 2011 and month < 9):
                net_path = path + 'net1m_{}-{}-01.txt'.format(str(year).zfill(2), str(month).zfill(2))
                timings[f'{year}-{month}'] = len(x)
                net1 = read_net(net_path)
                if time >= dt:
                    parse_nets(net0, net1, x, n, groups)
                    net0 = update_net0(net0, net1, time, dt)
                    update_na(na, net0, groups)
                    print('time: {}; n0_size={}'.format(time, net0.number_of_edges()))
                elif time == 0:
                    net0 = net1.copy()
                else:
                    net0 = update_net0(net0, net1, time, dt)
                    net_base = net0.copy()
                time += 1
    na = len(na['a']) / (len(na['a']) + len(na['b']))
    obs = get_last_obs(net0, groups)
    obs['na'] = na
    obs['N'] = net_base.number_of_nodes()
    obs['L'] = net_base.number_of_edges()

    x = {i: x_i for i, x_i in enumerate(x)}
    n = {i: n_i for i, n_i in enumerate(n)}
    if return_net:
        obs_0 = get_last_obs(net_base, groups)
        obs_0['N'] = net_base.number_of_nodes()
        obs_0['L'] = net_base.number_of_edges()
        return x, n, obs, net_base, obs_0
    elif return_times:
        return x, n, timings
    else:
        return x, n, obs


def plot_ranges(dt=12):
    fig, ax = plt.subplots()
    sas, sbs, cs = [], [], []
    for i in range(2002, 2011):
        x, n, obs = get_data_in_range(i, i+1, dt=dt)
        RF = rdf.RewireFit(x, n, .3)
        sa, sb, c = RF.solve()
        print(sa, sb, c)
        sas.append(sa); sbs.append(sb); cs.append(c)
    ax.plot(range(2003, 2012), sas, label=r'$s_a$')
    ax.plot(range(2003, 2012), sbs, label=r'$s_b$')
    ax.plot(range(2003, 2012), cs, label=r'$c$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Estimate')
    ax.set_title('Boards of Directors in Norway')
    fig.legend()
    #fig.tight_layout()
    fig.savefig('plots/temporal_dt{}_boards_directors.pdf'.format(dt))


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
        data = {'sas': sas, 'sbs': sbs, 'cs': cs, 'nas': nas, 'rhos': rhos, 'taas': taas, 'tbbs': tbbs}
        import pickle
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

    pred_ts = [gfp.rewire_fixed_points(c, na, sa, sb) for c, na, sa, sb in zip(cs, nas, sas, sbs)]
    ptaas = [pred_t[0][0] for pred_t in pred_ts]
    ptbbs = [pred_t[0][1] for pred_t in pred_ts]
    pred_cp = [gfp.rewire_fixed_points_density(c, na, sa, sb, rho*7) for c, na, sa, sb, rho in zip(cs, nas, sas, sbs, rhos)]
    pred_cpa = [x[1] for x in pred_cp]
    pred_cpb = [x[2] for x in pred_cp]

    obs_cpa = [gfp.cp_correlation_cont(paa, pbb, na, rho*7, 0.01)+np.random.uniform(-.01, .01) for paa, pbb, na, rho in zip(ptaas, ptbbs, nas, rhos)]
    obs_cpb = [gfp.cp_correlation_cont(pbb, paa, 1-na, rho*7, 0.01) for paa, pbb, na, rho in zip(ptaas, ptbbs, nas, rhos)]

    axs[1].plot(xvals, ptaas, '-', color='orangered', alpha=.5)
    axs[1].plot(xvals, ptbbs, '-', color='royalblue', alpha=.5)
    axs[1].plot(xvals, taas, '-', color='coral', alpha=.5)
    axs[1].plot(xvals, tbbs, '-', color='deepskyblue', alpha=.5)

    axs[1].plot(xvals, ptaas, 'o', label=r'$P^p_{aa}$', color='orangered')
    axs[1].plot(xvals, ptbbs, 'o', label=r'$P^p_{bb}$', color='royalblue')
    axs[1].plot(xvals, taas, 'x', label=r'$P^o_{aa}$', color='coral')
    axs[1].plot(xvals, tbbs, 'x', label=r'$P^o_{bb}$', color='deepskyblue')

    axs[2].plot(xvals, pred_cpa, 'o', label=r'$r^p$', color='palegreen')
    #axs[2].plot(xvals, pred_cpb, 'o', label=r'$r^p_B$', color='b')
    axs[2].plot(xvals, obs_cpa, 'x', label=r'$r^o$', color='slategrey')
    #axs[2].plot(xvals, obs_cpb, 'x', label=r'$r^o_B$', color='cyan')

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

# Main func
def plot_snapshots_evolution(dt=12, n_samp=5):
    fig, axs= plt.subplots(1, 5, figsize=(5*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    sas0, sbs0 = [], []
    #dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2011)] #, (2009, 2011)]
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
        pe.plot_rewire_predict(sa, sb, c, na, rho, P0=P0, P_obs=P_obs, r_steps=r_steps,
                ax=axs[i+1], title='{}-{}'.format(start, end+1), extra=extra, rewiring_metadata=rmeta)
        fig.savefig('boards/snapshot_prediction_{}.pdf'.format(dt))

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

# main func
def plot_snapshots_policy(var='c', factor=.5, dt=12):
    fig, axs= plt.subplots(1, 4, figsize=(4*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    sas0, sbs0 = [], []
    #dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2011)] #, (2009, 2011)]
    for i, (start, end) in enumerate(dates):
        x, n, obs, net_base, obs_0 = get_data_in_range(start, end, dt=dt, return_net=True)
        na = obs['na']
        N, L = obs_0['N'], obs_0['L']
        print(2*L/N**2)
        r_steps = len(x)
        RF = rdf.RewireFit(x, n, .3)
        sa, sb, c = RF.solve()
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        P0 = (obs_0['paa'], obs_0['pbb'])
        t_size = pe.gfp.rewire_steps(c, na, sa, sb, P0, N, L, 3)
        if t_size is not None:
            t_size = np.mean(t_size)
        else:
            t_size = 2500
        extra = r_steps/(end-start)
        rho = 2*L / (N*(N-1))
        P_obs = (obs['paa'], obs['pbb'])
        #rmetal = np.linspace(0, r_steps, end-start+1)
        #rmetat = range(start+1, end+2) #in this formulation end includes that year (check)
        #rmeta = [(x, y) for x, y in zip(rmetal, rmetat)]
        ts = [np.linspace(0, r_steps/t_size, 5000), np.linspace(0, 2*r_steps/t_size, 5000)]

        params = {'sa': sa, 'sb': sb, 'c': c, 'na': na}
        changes = [[var, (np.round(params[var], 2), factor_lims(params[var], factor))]]
        print(changes)
        pe.plot_rewire_policy(params, changes=changes, P0=P0, ax=axs[i], ts=ts, t_size=2500, rho=rho, title='{}-{}'.format(start, end+1))
        fig.savefig('boards/snapshot_policy_dt{}_var{}_factor{}.pdf'.format(dt, var, factor))

    fig.suptitle('Boards of Directors in Norway\n Policy: change {}'.format(var))

    fig.tight_layout()
    fig.savefig('boards/snapshot_policy_dt{}_var{}_factor{}.pdf'.format(dt, var, factor))

def factor_lims(val, factor):
    rfac = factor*val
    val = max([0, rfac])
    val = min([1, val])
    return np.round(val, 2)


def plot_snapshots_evolution_c0(dt=12):
    fig, axs= plt.subplots(1, 6, figsize=(6*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    for i, (start, end) in enumerate(dates):
        x, n, obs, net_base, obs_0 = get_data_in_range(start, end, dt=dt, return_net=True)
        na = obs['na']
        RF = rdf.RewireFit(x, n, na)
        sa, sb = RF.solve_c0()
        print(sa, sb)
        sas.append(sa); sbs.append(sb); nas.append(na)
        P = (obs_0['paa'], obs_0['pbb'])
        t = np.linspace(0, 50, 1000)
        ysol = gfp.rewire_path(c=0, sa=sa, sb=sb, na=na, P=P, t=t)
        ysol = p_to_t(ysol)

        dists = [np.sqrt((obs['taa'] - x[0])**2 + (obs['tbb']-x[1])**2) for x in ysol]
        xdist = np.argmin(dists)

        psim = gfp.rewire_simul_n(0, na, sa, sb, P, len(x), obs_0['N'], obs_0['L'], n_samp=5)
        tsim = p_to_t(psim)
        tsim = np.mean(tsim, axis=0)

        distsim = [np.sqrt((tsim[0] - x[0])**2 + (tsim[1]-x[1])**2) for x in ysol]
        xdistsim = np.argmin(distsim)

        axs[i+1].plot(t, ysol[:, 0], color='g', label='Predicted ' + r'$T_{aa}$')
        axs[i+1].plot(t, ysol[:, 1], color='b', label='Predicted ' + r'$T_{bb}$')
        axs[i+1].plot(t[xdistsim], tsim[0], 'x', label='Simulated '+ r'$T_{aa}$', color='g')
        axs[i+1].plot(t[xdistsim], tsim[1], 'x', label='Simulated '+ r'$T_{bb}$', color='b')
        axs[i+1].plot(t[xdist], obs['taa'], 'o', label='Observed '+ r'$T_{aa}$', color='g')
        axs[i+1].plot(t[xdist], obs['tbb'], 'o', label='Observed '+ r'$T_{bb}$', color='b')
    xvals = ['{}-{}'.format(start+1, end) for start, end in dates]
    axs[1].legend()
    axs[0].plot(xvals, sas, '-', color='g', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='b', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='k', alpha=.5)
    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='g')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='b')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='k')


    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Boards of Directors in Norway\n No Pref. Attch.')
    axs[0].legend()

    for i in range(5):
        axs[i+1].set_xlabel('Mean-field time')
        axs[i+1].set_ylabel('Estimate')
        axs[i+1].set_title('{}-{}'.format(dates[i][0]+1, dates[i][1]))

    fig.tight_layout()
    fig.savefig('plots/snapshot_evolution_c0_dt{}_boards_directors.pdf'.format(dt))

def p_to_t(sols):
    t = []
    for y in sols:
        paa, pbb = y
        pab = 1 - paa - pbb
        taa = 2*paa / (2*paa + pab)
        tbb = 2*pbb / (2*pbb + pab)
        t.append([taa, tbb])
    return np.array(t)


def get_data_in_range_grow(start=2002, end=2007, dt=12, return_net=False):
    net0 = None
    groups = parse_groups()
    x, n, na = [], [], {'a': set(), 'b': set()}
    time = 0
    for year in range(start, end + 1):
        for month in range(1, 13):
            if (year > 2002 and year < 2011) or (year == 2002 and month > 4) or (year == 2011 and month < 9):
                net_path = path + 'net1m_{}-{}-01.txt'.format(str(year).zfill(2), str(month).zfill(2))
                net1 = read_net(net_path)
                if time >= dt:
                    parse_nets_grow(net0, net1, x, n, groups)
                    net0 = update_net0_grow(net0, net1, time, dt)
                    update_na(na, net0, groups)
                    print('time: {}; n0_size={}'.format(time, net0.number_of_edges()))
                elif time == 0:
                    net0 = net1.copy()
                else:
                    net0 = update_net0_grow(net0, net1, time, dt)
                    net_base = net0.copy()
                time += 1
    na = len(na['a']) / (len(na['a']) + len(na['b']))
    obs = get_last_obs(net0, groups)
    obs['na'] = na

    x = {i: x_i for i, x_i in enumerate(x)}
    n = {i: n_i for i, n_i in enumerate(n)}
    if return_net:
        obs_0 = get_last_obs(net_base, groups)
        return x, n, obs, net_base, obs_0
    else:
        return x, n, obs


def update_net0_grow(net0, net1, time, dt):
    for (n1, n2) in net1.edges():
        net0.add_edge(n1, n2, t=time)
    return net0


def net_diff_grow(net, nnet, groups):
    x_i = []
    noadd = 0
    degs = Counter([v for k, v in net.degree()])

    for edge in nnet.edges():
        if not net.has_edge(*edge):
            et = edge_type(edge, groups, net, nnet)

            tgt_deg = net.degree(edge[1])
            if isinstance(tgt_deg, int):
                x = (et, tgt_deg, degs[tgt_deg])
                x_i.append(x)
            else:
                noadd += 1

    xcount = Counter(x_i)
    x_i = [[[k[0], k[1], k[2], v]] for k, v in xcount.items()]
    print('     +{}'.format(len(x_i)))
    print('     -{}'.format(noadd))
    return x_i

def net_tots_grow(net, groups):
    n_i = {'na': {}, 'nb': {}}
    for node in net:
        deg = net.degree(node)
        if groups[node] == '2':
            n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
        else:
            n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
    pa = sum([k*v for k, v in n_i['na'].items()])
    pb = sum([k*v for k, v in n_i['nb'].items()])
    ua = sum(n_i['na'].values())
    ub = sum(n_i['nb'].values())
    n = [pa, pb, ua, ub]
    return n

def parse_nets_grow(net0, net1, x, n, groups):
    n_ = net_tots_grow(net0, groups)
    x_i = net_diff_grow(net0, net1, groups)
    n_i = [n_ for _ in range(len(x_i))]

    x += x_i
    n += n_i

def plot_snapshots_evolution_grow(dt=12):
    fig, axs= plt.subplots(1, 6, figsize=(6*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    dates = [(2002, 2003), (2002, 2005), (2002, 2007), (2002, 2009), (2002, 2011)]
    add = [dt, dt*3, dt*5, dt*7, dt*9, dt*11]
    for i, (start, end) in enumerate(dates):
        x, n, obs, net_base, obs_0 = get_data_in_range_grow(start, end, dt=add[i], return_net=True)
        na = obs['na']
        GF = gdf.GrowthFit(x, n, .3)
        sa, sb, c = GF.solve()
        print(sa, sb, c)
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        P = (obs_0['paa'], obs_0['pbb'])
        t = np.linspace(0, 50, 1000)
        ysol = gfp.rewire_path(c=c, sa=sa, sb=sb, na=na, P=P, t=t)
        ysol = p_to_t(ysol)
        dists = [np.sqrt((obs['taa'] - x[0])**2 + (obs['tbb']-x[1])**2) for x in ysol]
        xdist = np.argmin(dists)
        # Convert ysol from P to T
        axs[i+1].plot(t, ysol[:, 0], color='g', label='Predicted ' + r'$T_{aa}$')
        axs[i+1].plot(t, ysol[:, 1], color='b', label='Predicted ' + r'$T_{bb}$')
        axs[i+1].plot(t[xdist], obs['taa'], 'x', label='Observed '+ r'$T_{aa}$', color='g')
        axs[i+1].plot(t[xdist], obs['tbb'], 'x', label='Observed '+ r'$T_{bb}$', color='b')
    xvals = ['{}-{}'.format(start+1, end) for start, end in dates]
    axs[1].legend()
    axs[0].plot(xvals, sas, '-', color='g', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='b', alpha=.5)
    axs[0].plot(xvals, cs, '-', color='r', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='k', alpha=.5)

    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='g')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='b')
    axs[0].plot(xvals, cs, 'o', label=r'$c$', color='r')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='k')


    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Boards of Directors in Norway\n Growth Estimated Parameters')
    axs[0].legend()

    for i in range(5):
        axs[i+1].set_xlabel('Mean-field time')
        axs[i+1].set_ylabel('Estimate')
        axs[i+1].set_title('{}-{}'.format(dates[i][0]+1, dates[i][1]))

    fig.tight_layout()
    fig.savefig('plots/snapshot_evolution_growth_dt{}_boards_directors.pdf'.format(dt))
