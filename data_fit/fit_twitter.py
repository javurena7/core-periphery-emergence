import numpy as np
import pandas as pd
import networkx as nx
import rewire_degree_fit as rdf
import growth_degree_fit as gdf
import matplotlib.pyplot as plt; plt.ion()
from collections import Counter, OrderedDict
import datetime as dt
import get_fixed_points as gfp
import os
import pickle
import plot_policies as pp
from copy import deepcopy


logpath = '../../../courses/twitter/Data/climate_twitter/twitter_fi_climate_edges_20200824-20201207.csv'

logpath = '../data/twitter/preprocess_samp.csv'

def read_logs(lpath=logpath, ttype='retweet', date0='2020-08-24', daten='2020-09-30', convert_timestamp=False):
    df = pd.read_csv(lpath, header=0)
    df = df[df.type==ttype]
    df = df[['source', 'target', 'timestamp']]
    if convert_timestamp:
        df['timestamp'] = pd.to_datetime(df.timestamp, infer_datetime_format=True)
    else:
        df.timestamp = df.timestamp / 1000
        df.timestamp = df['timestamp'].apply(dt.datetime.fromtimestamp)

    if date0:
        hpr = [int(x) for x in date0.split('-')]
        date0 = dt.datetime(hpr[0], hpr[1], hpr[2])

        df = df[df.timestamp > date0]

    if daten:
        hpr = [int(x) for x in daten.split('-')]
        daten = dt.datetime(hpr[0], hpr[1], hpr[2])
        df = df[df.timestamp < daten]
    #print(f'Number of elements: {df.shape[0]}')
    return df

def add_order(order, line, net):
    if net.has_edge(line.source, line.target):
        if order.get((line.source, line.target)):
            order[(line.source, line.target)] = line.timestamp
        else:
            order[(line.target, line.source)] = line.timestamp
    else:
        order[(line.source, line.target)] = line.timestamp
    return order


def directed_pmat(net):
    groups = parse_groups()
    labs = ['aa', 'ab', 'ba', 'bb']
    P = {l: 0 for l in labs}
    a = set(); b = set()
    for edge in net.edges():
        a1 = 'a' if edge[0] in groups else 'b'
        a2 = 'a' if edge[1] in groups else 'b'
        P[a1+a2] += 1
        node_to_gset(a, b, edge[0], a1)
        node_to_gset(a, b, edge[1], a2)
    tot = sum(P.values())
    rhos = get_dens(a, b, P, tot)
    P = {k: v/tot for k, v in P.items()}
    return P, rhos #, tot


def node_to_gset(a, b, node, grp):
    if grp == 'b':
        b.add(node)
    else:
        a.add(node)


def get_dens(a, b, P, tot):
    a = len(a); b = len(b)
    labs = ['aa', 'ab', 'ba', 'bb']
    rhos = {}
    for lab in labs:
        if lab[0] == lab[1]:
            den = eval(lab[0]) * (eval(lab[0])-1)
            rhos['r'+lab] = 2 * P[lab] / den
        else:
            den = eval(lab[0]) * eval(lab[1])
            rhos['r'+lab] = P[lab] / den
    return rhos



def get_net0(df, DT=60*60*24*7):
    net = nx.empty_graph(create_using=nx.DiGraph)
    line = df.iloc[0]
    i, dt = 1, 0
    time0 = line.timestamp
    net.add_edge(line.source, line.target, w=time0)
    order = OrderedDict()
    order[(line.source, line.target)] = time0
    while dt < DT:
        line = df.iloc[i]
        delta = line.timestamp - time0
        dt = delta.total_seconds()
        #print(dt)
        order = add_order(order, line, net)
        net.add_edge(line.source, line.target)
        i += 1
    df = df.iloc[i:]
    return net, df, order

def parse_groups():
    v_path = '../data/twitter/verified_accounts.txt'
    mp_path = '../../../courses/twitter/Data/twitter_mps.csv'
    verified = pd.read_csv(v_path, sep=' ', names=('idx', 'handle'))
    ver_ids = set(verified.idx)
    mps = pd.read_csv(mp_path)
    mp_ids = set(mps.twitter_id)
    groups = ver_ids.union(mp_ids)
    return groups

def edge_type(edge, groups):

    #g1 = groups.get(edge[0]
    #g2 = groups[edge[1]]
    a1 = 'a' if edge[0] in groups else 'b'
    a2 = 'a' if edge[1] in groups else 'b'
    return a1 + a2


def net_diff_idx(net, df, order, groups, DT, i):
    # Evolve network until we create a new link
    x_i = []
    add = 0
    degs = Counter([v for k, v in net.degree()])
    df_len = df.shape[0]
    while not x_i and i < df_len:
        line = df.iloc[i]
        edge = (line.source, line.target)
        if not net.has_edge(*edge):
            et = edge_type(edge, groups)
            try:
                tgt_deg = net.degree(edge[1])
            except:
                tgt_deg = ''
            if isinstance(tgt_deg, int):
                x_i.append(et)
                x_i.append(tgt_deg)
                x_i.append(degs[tgt_deg])
                net.add_edge(*edge)
                order = add_order(order, line, net)
            else:
                add += 1
                net.add_edge(*edge)
                order = add_order(order, line, net)
                i += 1
        else:
            #noadd += 1
            order = add_order(order, line, net)
            i += 1

    return [x_i], net, i

def net_tots(net, groups):
    n_i = {'na': {}, 'nb': {}}
    for node in net:
        deg = net.degree(node)
        if node in groups:
            n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
        else:
            n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
    pa = sum([k*v for k, v in n_i['na'].items()])
    pb = sum([k*v for k, v in n_i['nb'].items()])
    ua = sum(n_i['na'].values())
    ub = sum(n_i['nb'].values())
    n = [pa, pb, ua, ub]
    return n

def parse_nets(net0, df, order, x, n, groups, DT, i):
    n_ = net_tots(net0, groups)
    x_i, net0, i = net_diff_idx(net0, df, order, groups, DT, i)
    #print(f'order len: {len(order)}')
    n_i = [n_ for _ in range(len(x_i))]
    if x_i:
        x += x_i
        n += n_i
    return i

def update_net0(net0, order, DT):
    last = next(reversed(order.values()))
    rem_keys = []
    for k, v in order.items():
        delta = last - v
        t = delta.total_seconds()
        if t > DT:
            try:
                net0.remove_edge(*k)
                rem_keys.append(k)
            except:
                pass
    for k in rem_keys:
        order.pop(k)
    return net0

def update_na(na, net0, groups):
    for node in net0.nodes():
        #g = groups[node]
        if node in groups:
            na['a'].add(node)
        else:
            na['b'].add(node)

def get_last_obs(net0, groups):
    """
    Get Taa, Tbb, Paa, Pbb
    """
    paa, pbb, pab = 0, 0, 0
    n_a = {'a': set(), 'b': set()}
    for edge in net0.edges():
        a1 = 'a' if edge[0] in groups else 'b'
        a2 = 'a' if edge[1] in groups else 'b'

        if a1 == a2:
            if a1 == 'a':
                paa += 1
            else:
                pbb += 1
        else:
            pab += 1
        n_a[a1].add(edge[0])
        n_a[a2].add(edge[1])


    tot = paa + pab + pbb
    paa /= tot
    pbb /= tot
    pab /= tot

    taa = 2*paa / (2*paa + pab)
    tbb = 2*pbb / (2*pbb + pab)
    na = len(n_a['a']) / (len(n_a['a']) + len(n_a['b']))

    obs = {'paa': paa,
            'pab': pab,
            'pbb': pbb,
            'N': net0.number_of_nodes(),
            'L': net0.number_of_edges(),
            'na': na}

    return obs


def get_data_in_range(DT=60*60*24*7, date0='2020-08-24', daten='2020-09-01', ttype='retweet', return_net=False):
    dth = int(DT / 60*60)
    if ttype == 'retweet':
        path = f'twitter/{date0}_{daten}_dt{dth}.p'
    else:
        path = f'twitter/{date0}_{daten}_dt{dth}_tt{ttype}.p'

    if os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
        return data

    net0 = None
    groups = parse_groups()
    x, n, na = [], [], {'a': set(), 'b': set()}
    df = read_logs(logpath, ttype, date0, daten)
    net0, df, order = get_net0(df, DT)
    time = 0
    net_base = net0.copy()
    i = 0
    while i < df.shape[0]-2:
        i = parse_nets(net0, df, order, x, n, groups, DT, i)
        net0 = update_net0(net0, order, DT)
        update_na(na, net0, groups)
        time += 1
        print('time: {}; n0_size={}'.format(time, net0.number_of_edges()))
    na = len(na['a']) / (len(na['a']) + len(na['b']))
    obs = get_last_obs(net0, groups)
    obs['na'] = na
    obs0 = get_last_obs(net_base, groups)
    P0 = directed_pmat(net_base)
    P = directed_pmat(net0)
    x = {i: x_i for i, x_i in enumerate(x)}
    n = {i: n_i for i, n_i in enumerate(n)}
    data = {'x': x,
            'n': n,
            'obs0': obs0,
            'obs': obs,
            'P': P,
            'P0': P0}
    pickle.dump(data, open(path, 'wb'))
    return data


def plot_policies(DT=60*60*24*1, dates=[], ttype='retweet'):

    fig, axs= plt.subplots(4, 3, figsize=(4*3, 3*3))
    fdata = {k: [] for k in ['c', 'sa', 'sb', 'na', 'paa', 'pbb', 'rsteps', 'pab', 'pba', 'rho', 't_size', 'paa0', 'pab0', 'pba0', 'pbb0']}
    #sas, sbs, cs, nas, taas, tbbs = [], [], [], [], [], []
    if not dates:
        dates = [('2020-08-24', '2020-09-15'), ('2020-10-02', '2020-10-10'), ('2020-11-15', '2020-12-03')]
    #dates = [('2020-08-24', '2020-08-31'), ('2020-09-07', '2020-09-16'), ('2020-11-05', '2020-11-15')]
    #dates = [('2020-08-24', '2020-08-26'), ('2020-09-07', '2020-09-09'), ('2020-10-02', '2020-10-4'), ('2020-10-15', '2020-10-17'), ('2020-11-15', '2020-11-17')]
    dth = int(DT / (3600)) if DT / 3600 >= 1 else DT
    fdates = '.'.join([x.replace('2020-', '').replace('2021-', '') for y in dates for x in y])
    fpath = f'twitter/fdata_dt{dth}_{ttype}_{fdates}.p'

    if not os.path.exists(fpath):
        for start, end in dates:
            data = get_data_in_range(DT=DT, date0=start, daten=end, ttype=ttype)
            x = data['x']; n = data['n']; rsteps = len(x)
            P = data['P'][0]; P0 = data['P0'][0]
            obs = data['obs']; paa = P['aa']; pbb = P['bb']
            obs0 = data['obs0']
            pab = P['ab']; pba = P['ba']
            paa0 = P0['aa']; pbb0 = P0['bb']; pab0=P0['ab']; pba0 = P0['ba']

            paa = paa / (paa + pba + pbb); paa0 = paa0 / (paa0 + pba0 + pbb0)
            pbb = pbb / (paa + pba + pbb); pbb0 = pbb0 / (pbb0 + pba0 + paa0)

            patch = len(x)
            x.pop(patch - 1)
            na = obs['na']
            rho = 2 * obs['L'] / (obs['N'] * (obs['N'])-1)
            RF = rdf.RewireFit(x, n, na)
            sa, sb, c = RF.solve()
            print(sa, sb, c)
            t_size = gfp.rewire_steps(c, na, sa, sb, (paa0, pbb0), obs0['N'], obs0['L'], 5)
            if t_size is not None:
                t_size = np.mean(t_size)
            else:
                t_size = 2500
            for lab in fdata.keys():
                fdata[lab].append(eval(lab))
        fdata['dates'] = dates
        pickle.dump(fdata, open(fpath, 'wb'))
    else:
        fdata = pickle.load(open(fpath, 'rb'))

    for i, (start, end) in enumerate(dates):
        P0 = fdata['paa0'][i], fdata['pbb0'][i]
        P = fdata['paa'][i], fdata['pbb'][i]
        fx_params = {'c': fdata['c'][i],
                'na': fdata['na'][i],
                'sa': fdata['sa'][i],
                'sb': fdata['sb'][i]}
        rho = fdata['rho'][i]
        for j, x in enumerate(['c', 'na', 'sa', 'sb']):
            params = deepcopy(fx_params)
            pol_params = deepcopy(fx_params)
            obs_0 = {'paa': P0[0], 'pbb': P0[1], x: params.pop(x)}
            obs_n = {'paa': P[0], 'pbb': P[1]}

            legend = True if (i == 3) and (j == 0) else False
            print(f'getting {x}')
            pp.plot_fullrange_policy_rewire(x, params, rho, obs_0=obs_0, obs_n=obs_n,
                    ax=axs[j][i], nsize=25, legend=legend)

    #start, enddates):
    axs[0][i].set_title('{}-\n{}'.format(start, end))
    for j in range(4):
        axs[j].spines['top'].set_visible(False)
        axs[j].spines['right'].set_visible(False)
    dthh = int(dth / 24)
    fig.suptitle(f'Twitter parameter interventions \n DT={dthh} days')
    fig.tight_layout()
    fig.savefig(f'twitter/policy_agg_{ttype}_dt{dth}_{fdates}.pdf')

    return fdata


def plot_policy_aggregate(DT=60*60*24*1, dates=[], ttype='retweet'):
    """
    Plot policies for one aggregate fit over a large dataset
    """
    fig, axs= plt.subplots(1, 4, figsize=(4*3, 3))
    fdata = {k: [] for k in ['c', 'sa', 'sb', 'na', 'paa', 'pbb', 'rsteps', 'pab', 'pba', 'rho', 't_size', 'paa0', 'pab0', 'pba0', 'pbb0']}
    #sas, sbs, cs, nas, taas, tbbs = [], [], [], [], [], []
    if not dates:
        dates = ['2020-02-24', '2020-12-30']
    #dates = [('2020-08-24', '2020-08-31'), ('2020-09-07', '2020-09-16'), ('2020-11-05', '2020-11-15')]
    #dates = [('2020-08-24', '2020-08-26'), ('2020-09-07', '2020-09-09'), ('2020-10-02', '2020-10-4'), ('2020-10-15', '2020-10-17'), ('2020-11-15', '2020-11-17')]
    dth = int(DT / (3600)) if DT / 3600 >= 1 else DT
    #fdates = '.'.join([x.replace('2020-', '').replace('2021-', '') for y in dates for x in y])
    fdates = '.'.join(dates)
    fpath = f'twitter/fdata_dt{dth}_{ttype}_{fdates}.p'
    start, end = dates
    if not os.path.exists(fpath):
        data = get_data_in_range(DT=DT, date0=start, daten=end, ttype=ttype)
        x = data['x']; n = data['n']; rsteps = len(x)
        P = data['P'][0]; P0 = data['P0'][0]
        obs = data['obs']; paa = P['aa']; pbb = P['bb']
        obs0 = data['obs0']
        pab = P['ab']; pba = P['ba']
        paa0 = P0['aa']; pbb0 = P0['bb']; pab0=P0['ab']; pba0 = P0['ba']

        paa = paa / (paa + pba + pbb); paa0 = paa0 / (paa0 + pba0 + pbb0)
        pbb = pbb / (paa + pba + pbb); pbb0 = pbb0 / (pbb0 + pba0 + paa0)

        patch = len(x)
        x.pop(patch - 1)
        na = obs['na']
        rho = 2 * obs['L'] / (obs['N'] * (obs['N'])-1)
        RF = rdf.RewireFit(x, n, na)
        sa, sb, c = RF.solve()
        print(sa, sb, c)
        t_size = gfp.rewire_steps(c, na, sa, sb, (paa0, pbb0), obs0['N'], obs0['L'], 5)
        if t_size is not None:
            t_size = np.mean(t_size)
        else:
            t_size = 2500
        for lab in fdata.keys():
            fdata[lab].append(eval(lab))
        fdata['dates'] = dates
        pickle.dump(fdata, open(fpath, 'wb'))
    else:
        fdata = pickle.load(open(fpath, 'rb'))

    i = 0 #for i, (start, end) in enumerate(dates):
    P0 = fdata['paa0'][i], fdata['pbb0'][i]
    P = fdata['paa'][i], fdata['pbb'][i]
    fx_params = {'c': fdata['c'][i],
            'na': fdata['na'][i],
            'sa': fdata['sa'][i],
            'sb': fdata['sb'][i]}
    rho = fdata['rho'][i]
    for j, x in enumerate(['c', 'na', 'sa', 'sb']):
        params = deepcopy(fx_params)
        pol_params = deepcopy(fx_params)
        obs_0 = {'paa': P0[0], 'pbb': P0[1], x: params.pop(x)}
        obs_n = {'paa': P[0], 'pbb': P[1]}

        legend = True if (i == 3) and (j == 0) else False
        print(f'getting {x}')
        pp.plot_fullrange_policy_rewire(x, params, rho, obs_0=obs_0, obs_n=obs_n, ax=axs[j], nsize=25, legend=legend)

    axs[0].set_title('{}-\n{}'.format(start, end))
    for j in range(4):
        axs[j].spines['top'].set_visible(False)
        axs[j].spines['right'].set_visible(False)
    dthh = int(dth / 24)
    fig.suptitle(f'Twitter parameter interventions \n DT={dthh} days')
    fig.tight_layout()
    fig.savefig(f'twitter/policy_{ttype}_dt{dth}_{fdates}.pdf')

    return fdata


"""
    xvals = [start for start, end in dates]
    axs[0].plot(xvals, sas, '-', color='g', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='b', alpha=.5)
    axs[0].plot(xvals, cs, '-', color='r', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='k', alpha=.5)

    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='g')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='b')
    axs[0].plot(xvals, cs, 'o', label=r'$c$', color='r')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='k')

    pred_ts = [gfp.fixed_points(c, na, sa, sb) for c, na, sa, sb in zip(cs, nas, sas, sbs)]
    ptaas = [pred_t[0][0] for pred_t in pred_ts]
    ptbbs = [pred_t[0][1] for pred_t in pred_ts]
    axs[1].plot(xvals, ptaas, '-', color='g', alpha=.5)
    axs[1].plot(xvals, ptbbs, '-', color='b', alpha=.5)
    axs[1].plot(xvals, taas, '-', color='lime', alpha=.5)
    axs[1].plot(xvals, tbbs, '-', color='cyan', alpha=.5)

    axs[1].plot(xvals, ptaas, 'o', label='Predicted ' + r'$P_{aa}$', color='g')
    axs[1].plot(xvals, ptbbs, 'o', label='Predicted ' + r'$P_{bb}$', color='b')
    axs[1].plot(xvals, taas, 'o', label='Observed ' + r'$P_{aa}$', color='lime')
    axs[1].plot(xvals, tbbs, 'o', label='Observed ' + r'$P_{bb}$', color='cyan')

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Climate Twitter in Finland\n Estimated Parameters')
    axs[0].legend()

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Estimate')
    axs[1].set_title('Climate Twitter in Finland\n Estimated T-matrix')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig('twitter/snapshot_dt{}_climate_twitter_{}.pdf'.format(DT, ttype))
"""

def p_to_t(sols):
    t = []
    for y in sols:
        paa, pbb = y
        pab = 1 - paa - pbb
        taa = 2*paa / (2*paa + pab)
        tbb = 2*pbb / (2*pbb + pab)
        t.append([taa, tbb])
    return np.array(t)


def plot_snapshots_evolution(DT=60*60*24*2, ttype='retweet', n_samp=5):
    fig, axs= plt.subplots(1, 4, figsize=(4*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    sas0, sbs0 = [], []
    dates = [('2020-08-24', '2020-09-15'), ('2020-10-02', '2020-10-10'), ('2020-11-15', '2020-12-03')]
    dates = [('2020-08-24', '2020-08-30'), ('2020-10-02', '2020-10-08'), ('2020-11-15', '2020-11-21')]
    for i, (start, end) in enumerate(dates):
        x, n, obs, net_base, obs_0 = get_data_in_range(DT=DT, date0=start, daten=end, ttype=ttype, return_net=True)
        patch = len(x)
        x.pop(patch - 1)
        na = obs['na']
        RF = rdf.RewireFit(x, n, .05)
        sa, sb, c = RF.solve()
        RF = rdf.RewireFit(x, n, .05)
        sa0, sb0 = RF.solve_c0()

        print(sa, sb, c)
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        sas0.append(sa0); sbs0.append(sb0)
        P = (obs_0['paa'], obs_0['pbb'])
        t = np.linspace(0, 50, 1000)

        ysol = gfp.rewire_path(c=c, sa=sa, sb=sb, na=na, P=P, t=t)
        ysol = p_to_t(ysol)

        dists = [np.sqrt((obs['taa'] - x[0])**2 + (obs['tbb']-x[1])**2) for x in ysol]
        xdist = np.argmin(dists)

        ysol0 = gfp.rewire_path(c=0, sa=sa0, sb=sb0, na=na, P=P, t=t)
        ysol0 = p_to_t(ysol0)

        psim = gfp.rewire_simul_n(c, na, sa, sb, P, len(x), obs_0['N'], obs_0['L'], n_samp=n_samp)
        tsim = p_to_t(psim)
        tsim = np.mean(tsim, axis=0)

        distsim = [np.sqrt((tsim[0] - x[0])**2 + (tsim[1]-x[1])**2) for x in ysol]
        xdistsim = np.argmin(distsim)

        axs[i+1].plot(t, ysol[:, 0], color='g', label='Predicted ' + r'$T_{aa}$')
        axs[i+1].plot(t, ysol[:, 1], color='b', label='Predicted ' + r'$T_{bb}$')

        axs[i+1].plot(t, ysol0[:, 0], '--', color='g')
        axs[i+1].plot(t, ysol0[:, 1], '--', color='b')

        axs[i+1].plot(t[xdistsim], tsim[0], 'x', label='Simulated '+ r'$T_{aa}$', color='g')
        axs[i+1].plot(t[xdistsim], tsim[1], 'x', label='Simulated '+ r'$T_{bb}$', color='b')
        axs[i+1].plot(t[xdist], obs['taa'], 'o', label='Observed '+ r'$T_{aa}$', color='g')
        axs[i+1].plot(t[xdist], obs['tbb'], 'o', label='Observed '+ r'$T_{bb}$', color='b')
        fig.savefig('plots/snapshot_evolution_dt{}_climate_twitter_{}_5days.pdf'.format(DT, ttype))
    xvals = ['{}-\n{}'.format(start, end) for start, end in dates]
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
    axs[0].set_title('Climate Twitter in Finland\n Estimated Parameters')
    axs[0].legend()

    for i in range(3):
        axs[i+1].set_xlabel('Mean-field time')
        axs[i+1].set_ylabel('Estimate')
        axs[i+1].set_title('Evol. {}\n-{}'.format(dates[i][0], dates[i][1]))

    fig.tight_layout()
    fig.savefig('plots/snapshot_evolution_dt{}_climate_twitter_{}_5days.pdf'.format(DT, ttype))

def plot_snapshots_evolution_growth(gdt=60*60*24, ttype='retweet'):
    fig, axs= plt.subplots(1, 4, figsize=(4*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    dates = [('2020-08-24', '2020-09-15'), ('2020-10-02', '2020-10-23'), ('2020-11-15', '2020-12-07')]
    dates = [('2020-08-24', '2020-09-27'), ('2020-10-02', '2020-10-05'), ('2020-11-15', '2020-11-18')]
    for i, (start, end) in enumerate(dates):
        df = read_logs(logpath, ttype=ttype, date0=start, daten=end)
        net, df, _ = get_net0(df, DT=60*60*12)
        print(f'net0: {start}; first_fit: {df.timestamp.iloc[0]}')
        x, n, obs_0, obs = sort_sources_growth(net, df, gdt)
        na = obs['na']
        GF = gdf.GrowthFit(x, n, na)
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
        axs[i+1].plot(t[xdist], obs['taa'], 'x', label='Observed ' + r'$T_{aa}$', color='g')
        axs[i+1].plot(t[xdist], obs['tbb'], 'x', label='Observed ' + r'$T_{bb}$', color='b')
        fig.savefig('plots/snapshot_evolution_growth_gdt{}_climate_twitter_{}_2days.pdf'.format(gdt, ttype))
    xvals = ['{}-\n{}'.format(start, end) for start, end in dates]
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
    axs[0].set_title('Climate Twitter in Finland\n Estimated Parameters')
    axs[0].legend()

    for i in range(3):
        axs[i+1].set_xlabel('Mean-field time')
        axs[i+1].set_ylabel('Estimate')
        axs[i+1].set_title('{} -\n{}'.format(dates[i][0], dates[i][1]))

    fig.tight_layout()
    fig.savefig('plots/snapshot_evolution_growth_gdt{}_climate_twitter_{}_2days.pdf'.format(gdt, ttype))


def sort_sources_growth(net, df, gdt):
    """
    For all nodes who are new in net, get all their new connections in a range dt from the first activity
    """
    groups = parse_groups()
    gdt = dt.timedelta(seconds=gdt)
    df = df.set_index('source')
    deg_dist = Counter([v for k, v in net.degree()])
    obs0 = get_last_obs(net, groups)
    x = {}
    i = 0
    n = {i: net_tots(net, groups)}
    deg_dist = {}
    for (src, line) in df.iterrows():
        if True: #src not in net:
            dftmp = df.loc[src]
            try:
                base = dftmp.iloc[0].timestamp
                dftmp = dftmp[dftmp.timestamp - base < gdt]
                targets = list(dftmp.target)
            except:
                targets = [dftmp.target]
            xt, nt = [], []
            for tgt in targets:
                et = edge_stats(src, tgt, net, groups, deg_dist)
                nt = net_tots(net, groups)
                xt.append(et)
                try:
                    tgt_deg = net.degree(tgt)
                except:
                    tgt_deg = 1

                deg_dist[tgt_deg] = deg_dist.get(tgt_deg, 0) + 1
                net.add_edge(src, tgt)
            src_deg = len(xt)
            deg_dist[src_deg] = deg_dist.get(src_deg, 0) + 1
            xcount = Counter(xt)
            xt = [[k[0], k[1], k[2], v] for k, v in xcount.items()]
            x[i] = xt
            n[i+1] = nt
            i += 1
        else:
            tgt = line.target
            net.add_edge(src, tgt)
            tgt_deg = net.degree(tgt)
            deg_dist[tgt_deg] = deg_dist.get(tgt_deg, 0) + 1
            src_deg = net.degree(src)
            deg_dist[src_deg] = deg_dist.get(src_deg, 0) + 1

    obs = get_last_obs(net, groups)

    return x, n, obs0, obs


def edge_stats(src, tgt, net, groups, deg_dist):
    etyp = edge_type((src, tgt), groups)
    try:
        tgt_deg = net.degree(tgt)
    except:
        tgt_deg = 1
    n_k = deg_dist.get(tgt_deg, 1)
    return (etyp, tgt_deg, n_k)

def get_loglik_fit(DT=60*60*24*1, dates=[], ttype='retweet'):
    """
    Get fit statistics, including loglik evaluated at optimum with and without c param
    """
    fdata = {} #{k: [] for k in ['c', 'sa', 'sb', 'na', 'paa', 'pbb', 'pab', 'pba', 'rho', 'paa0', 'pab0', 'pba0', 'pbb0', 'rhoa', 'rhob', 'rhoab', 'rhoba', 'sa0', 'sb0']}
    if not dates:
        dates = ['2020-02-24', '2020-12-30']
    dth = int(DT / (3600)) if DT / 3600 >= 1 else DT
    #fdates = '.'.join([x.replace('2020-', '').replace('2021-', '') for y in dates for x in y])
    fdates = '_'.join(dates)
    fpath = f'twitter/onion_fit/opt_fitdata.p'
    #TODO: dont do different fpaths, only one that is read and results added
    start, end = dates
    try:
        fdata = pickle.load(open(fpath, 'rb'))
    except:
        fdata = {}
    date_id = f'{fdates}_{ttype}_{dth}'

    if not date_id in fdata:
        fdat = {}
        data = get_data_in_range(DT=DT, date0=start, daten=end, ttype=ttype)
        x = data['x']; n = data['n']

        patch = len(x)
        x.pop(patch - 1); n.pop(patch-1)
        na = data['obs']['na']
        #rho = 2 * obs['L'] / (obs['N'] * (obs['N'])-1)
        RF = rdf.RewireFit(x, n, na)
        opt = RF.solve()
        llik = RF.loglik(opt)
        print(opt)

        opt0 = RF.solve_c0()
        llik_c0 = RF.loglik_c0(opt0)

        print(opt0)

        sol = {'sa': opt[0], 'sb': opt[1], 'c': opt[2],
                'sa0': opt0[0], 'sb0': opt0[1], 'llik': llik, 'llik_c0': llik_c0,
                'chi': 2*(llik - llik_c0)}
        fdat['obs'] = data['obs']
        fdat['obs0'] = data['obs0']
        fdat['rho'] = data['P'][1]
        fdat['rho0'] = data['P0'][1]
        fdat['fit'] = sol
        fdat['dates'] = dates
        fdat['DT'] = DT
        fdat['ttype'] = ttype
        fdata[date_id] = fdat
        pickle.dump(fdata, open(fpath, 'wb'))
    return fdata


if __name__ == '__main__':
    import argparse
    outpath = 'twitter/run_pairs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=int, default=60*60*24*7)
    parser.add_argument('--date0', type=str, default='2020-02-15')
    parser.add_argument('--daten', type=str, default='2021-01-04')

    args = parser.parse_args()

    dates = [args.date0, args.daten]
    get_loglik_fit(DT=args.dt, dates=dates, ttype='retweet')




