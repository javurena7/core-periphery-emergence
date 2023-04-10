import numpy as np
import networkx as nx
import rewire_degree_fit as rdf
import matplotlib.pyplot as plt; plt.ion()
from collections import Counter
from itertools import product
from pandas import read_csv
import get_fixed_points as gfp
from os import path as ospath

path = '../data/airport/db1b/' #Path where data files are located

class RouteEvol(object):
    """
    Preprocessing for airport routes evolution data (DB1B Product from the US Bureau of Transportation Statistics) in a range of "years", where the observation is of length "dt" (in quarters of a year)
    rewire_net: produces two dicts that contain network evolution data
    """
    def __init__(self, a=(1, 9), b=(2, 3, 4, 5, 6, 7, 8), years=(1993, 2021), dt=4):
        self.a, self.b = a, b
        self.years, self.dt = years, dt
        self.x, self.n = {}, {}
        self.routes = []
        self.missing_airports = set()
        self.missing_states = set()
        self.i = 0
        self._quarter_iter = product(range(*years), range(1, 5))
        self._get_regions()
        self._init_nets()

    def _get_regions(self):
        df = read_csv('../data/airport/db1b/us_regions.txt', dtype={"State": str, "Region": str, "Code": int})
        self.state_region = dict(df[['State', 'Code']].values)
        self.state_region_names = dict(df[['Code', 'Region']].values)

    def _valid_routes(self, nroutes):
        """
        Add only routes we can classify to the network
        """
        routes = set()
        for rte in nroutes:
            src = self.get_group(rte[0])
            tgt = self.get_group(rte[1])
            if src and tgt:
                routes.add(rte)
        return routes

    def _init_nets(self):
        self.airport_ids = {}
        for _ in range(self.dt):
            year, qrt = next(self._quarter_iter)
            croutes, airport_ids = self.read_db(year, qrt)
            croutes = self._valid_routes(croutes)
            self.routes = [x.difference(croutes) for x in self.routes]
            self.routes.append(croutes)
            self.airport_ids.update(airport_ids)
        self.net = nx.empty_graph()
        self.net.add_edges_from(set().union(*self.routes))

    def read_db(self, year, quarter):
        routes = set()
        airport_ids = {}
        fpath = path + 'DB1BCoupon_{}_{}.csv'.format(year, quarter)
        with open(fpath, 'r') as r:
            header = r.readline()
            line = r.readline()
            lroutes = 0
            while line:
                try:
                    line = line.split(',')
                    route = (line[1], line[6])
                    routes.add(route)
                    if len(routes) > lroutes: #MktID(2,7), IATA(4,8), State(5,9)
                        airport_ids[line[1]] = line[5].strip('"')
                        airport_ids[line[6]] = line[9].strip('"')
                except:
                    print('Error:')
                    print(line)
                line = r.readline()
        return routes, airport_ids

    def rewire_net(self):
        """
        Run full rewire algorithm including all instances of update_net
        """
        try:
            yr, qrt = next(self._quarter_iter)
        except StopIteration:
            raise("Need additional final year for this value of dt // Restart iterator")
        while yr:
            self.update_net(yr, qrt)
            try:
                yr, qrt = next(self._quarter_iter)
            except StopIteration:
                yr = None


    def update_net(self, yr, qr):
        print('Updating net: year {} - quarter {}'.format(yr, qr))
        routes, airport_ids = self.read_db(yr, qr)

        #Find routes that will be eliminated (not "renewed" in the new quarter)
        self.routes = [x.difference(routes) for x in self.routes]

        #Find routes not present in the current network
        new_routes_all = routes.difference(set().union(*self.routes))
        new_routes_subnet = self.update_evol_stats(new_routes_all)
        self.net.remove_edges_from(self.routes.pop(0))
        self.net.add_edges_from(new_routes_subnet)
        self.routes.append(routes)

    def update_evol_stats(self, nroutes):
        xt = []
        nt = []
        new_routes_subnet = set()
        group_degs = self.get_group_degs()
        pa_tot = sum([k*v for k, v in group_degs['a'].items()])
        pb_tot = sum([k*v for k, v in group_degs['b'].items()])
        ua_tot = sum(group_degs['a'].values())
        ub_tot = sum(group_degs['b'].values())
        #x_i = [linktype, tgtdeg, n_k, x_k(times deg k was selected)]
        #n_i = [pa_tot, pb_tot, ua_tot, ub_tot]
        for rte in nroutes:
            src_grp = self.get_group(rte[0])
            tgt_grp = self.get_group(rte[1])
            if src_grp and tgt_grp:

                tgt_k = self.net.degree(rte[1])
                if not isinstance(tgt_k, int):
                    tgt_k = 1
                n_k = group_degs[tgt_grp][tgt_k]

                x_i = (src_grp + tgt_grp, tgt_k, n_k)
                n_i = [pa_tot, pb_tot, ua_tot, ub_tot]
                xt.append(x_i)
                nt.append(n_i)
                new_routes_subnet.add(rte)
        xi_counts = Counter(xt)
        xt = [[x[0], x[1], x[2], v] for x, v in xi_counts.items()]

        for xi, ni in zip(xt, nt):
            self.x[self.i] = xi
            self.n[self.i] = ni
            self.i += 1
        return new_routes_subnet

    def get_group(self, airport):
        state = self.airport_ids.get(airport, 0)
        if state == 0:
            self.missing_airports.add(airport)
        region = self.state_region.get(state, 0)
        if region == 0 and state != 0:
            self.missing_states.add(state)
        if region in self.a:
            return 'a'
        elif region in self.b:
            return 'b'
        else:
            return ''

    #def get_group(self, airport): # ONLY when using hubs
    #    if airport in self.a:
    #        return 'a'
    #    else:
    #        return 'b'

    def get_group_degs(self):
        #grp_ks = {'a': {}, 'b': {}}
        a_cnt = []
        b_cnt = []

        for nd, deg in self.net.degree():
            grp = self.get_group(nd)
            if grp == 'a':
                a_cnt.append(deg)
            elif grp == 'b':
                b_cnt.append(deg)
        grp_ks = {'a': Counter(a_cnt), 'b': Counter(b_cnt)}
        return grp_ks

    def get_net_cp_statistics(self):
        """
        Get Taa, Tbb, Paa, Pbb
        """
        laa, lbb, lab = 0, 0, 0
        grps = {'a': set(), 'b': set(), '': set()}
        for edge in self.net.edges():
            g1 = self.get_group(edge[0])
            g2 = self.get_group(edge[1])
            if g1 == g2:
                if g1 == 'a':
                    laa += 1
                else:
                    lbb += 1
            else:
                lab += 1
            grps[g1].add(edge[0])
            grps[g2].add(edge[1])
        tot = laa + lab + lbb

        N = len(grps['a']) + len(grps['b'])
        self.obs = {'L': tot,
                'laa': laa,
                'lab': lab,
                'lbb': lbb,
                'N': N,
                'na': len(grps['a']) / N if N > 0 else 0}

        return self.obs

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


def plot_snapshots(dt=12):
    fig, axs= plt.subplots(1, 2)
    sas, sbs, cs, nas, taas, tbbs = [], [], [], [], [], []
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    for start, end in dates:
        x, n, obs = get_data_in_range(start, end, dt=dt)
        na = obs['na']
        RF = rdf.RewireFit(x, n, .3)
        sa, sb, c = RF.solve()
        print(sa, sb, c)
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        taas.append(obs['taa']), tbbs.append(obs['tbb'])
    xvals = ['{}-{}'.format(start+1, end) for start, end in dates]
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

    axs[1].plot(xvals, ptaas, 'o', label='Predicted ' + r'$T_{aa}$', color='g')
    axs[1].plot(xvals, ptbbs, 'o', label='Predicted ' + r'$T_{bb}$', color='b')
    axs[1].plot(xvals, taas, 'o', label='Observed ' + r'$T_{aa}$', color='lime')
    axs[1].plot(xvals, tbbs, 'o', label='Observed ' + r'$T_{bb}$', color='cyan')

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Boards of Directors in Norway\n Estimated Parameters')
    axs[0].legend()

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Estimate')
    axs[1].set_title('Boards of Directors in Norway\n Estimated T-matrix')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig('plots/snapshot_dt{}_boards_directors.pdf'.format(dt))


def plot_snapshots_evolution(dt=12, n_samp=5):
    fig, axs= plt.subplots(1, 5, figsize=(5*3, 3), sharey=True)
    sas, sbs, cs, nas = [], [], [], []
    sas0, sbs0 = [], []
    #dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2009), (2009, 2011)]
    dates = [(2002, 2003), (2003, 2005), (2005, 2007), (2007, 2011)] #, (2009, 2011)]
    for i, (start, end) in enumerate(dates):
        x, n, obs, net_base, obs_0 = get_data_in_range(start, end, dt=dt, return_net=True)
        na = obs['na']
        RF = rdf.RewireFit(x, n, .3)
        sa, sb, c = RF.solve()
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
        fig.savefig('plots/snapshot_evolution_dt{}_boards_directors_4dates.pdf'.format(dt))

    xvals = ['{}-\n{}'.format(start+1, end) for start, end in dates]
    axs[1].legend()
    axs[0].plot(xvals, sas, '-', color='g', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='b', alpha=.5)
    axs[0].plot(xvals, sas0, '--', color='g', alpha=.5)
    axs[0].plot(xvals, sbs0, '--', color='b', alpha=.5)
    axs[0].plot(xvals, cs, '-', color='r', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='k', alpha=.5)

    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='g')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='b')
    axs[0].plot(xvals, cs, 'o', label=r'$c$', color='r')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='k')


    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Boards of Directors in Norway\n Estimated Parameters')
    axs[0].legend()

    for i in range(4):
        axs[i+1].set_xlabel('Mean-field time')
        axs[i+1].set_ylabel('Estimate')
        axs[i+1].set_title('Evolution \n{}-{}'.format(dates[i][0]+1, dates[i][1]))

    fig.tight_layout()
    fig.savefig('plots/snapshot_evolution_dt{}_boards_directors_4dates.pdf'.format(dt))


def p_to_t(sols):
    t = []
    for y in sols:
        paa, pbb = y
        pab = 1 - paa - pbb
        taa = 2*paa / (2*paa + pab)
        tbb = 2*pbb / (2*pbb + pab)
        t.append([taa, tbb])
    return np.array(t)

def run_combinations(r):
    """
    For group A define all combinations of size r, group B is the compliment
    """
    from itertools import combinations
    regions = list(range(1, 10))

    for a in combinations(regions, r):
        b = set(regions).difference(a)
        print('Running {} {}'.format(a, b))
        #full_run(a, b, (1993, 1995), dt=4)
        full_run(a, b, (2010, 2012), dt=4)

def run_subsets():
    from itertools import chain, combinations
    major_regions = {1: [1, 2],
            2: [3, 4],
            3: [5, 6, 7],
            4: [8, 9]}
    i = 1
    for sbset in chain.from_iterable(combinations([1,2,3,4], r) for r in range(4)):
        lsbset = len(sbset)
        if (lsbset > 0) and (lsbset < 4) and i < 8:
            b = set(range(1, 5)).difference(sbset)
            a = [v for k in sbset for v in major_regions[k]]
            b = [v for k in b for v in major_regions[k]]
            print('Running {} {}'.format(a, b))
            i += 1
            full_run(a, b, (1993, 1995), dt=4)
            full_run(a, b, (2019, 2021), dt=4)

def run_pairs(outpath='airport/results_pairs.txt'):
    """
    Run all pairwise combinations based on the 9 major regions
    """
    from itertools import combinations
    regions = list(range(1, 10))

    for a, b in combinations(regions, 2):
        print('Running {} {}'.format(a, b))
        #full_run(a, b, (1993, 1995), dt=4)
        full_run([a], [b], (2010, 2012), dt=4, outpath=outpath)


def obs_to_line(obs):
    hd = ['L', 'laa', 'lab', 'lbb', 'N', 'na']
    line = '|'.join([str(obs[h]) for h in hd])
    return line

#Imp
def full_run(a, b, years, dt, outpath='airport/results.txt'):
    RE = RouteEvol(a, b, years, dt)
    obs0 = RE.get_net_cp_statistics()
    RE.rewire_net()
    obs = RE.get_net_cp_statistics()
    RF = rdf.RewireFit(RE.x, RE.n, obs0['na'])

    sol = RF.solve_randx0(5)
    sol0 = RF.solve_c0()
    astr, bstr = [str(x) for x in a], [str(x) for x in b]
    lg = '{}|{}'.format(''.join(astr), ''.join(bstr))
    ly = '{}|{}|{}'.format(years[0], years[1], dt)
    l0 = obs_to_line(obs0)
    l = obs_to_line(obs)
    ls = '{}|{}|{}'.format(*sol)
    ls0 = '{}|{}'.format(*sol0)

    header='a|b|y0|yn|dt|L0|laa0|lab0|lbb0|N0|na0|L|laa|lab|lbb|N|na|sa|sb|c|sa0|sb0\n'
    if not ospath.exists(outpath):
        with open(outpath, 'w') as w:
            w.write(header)
    with open(outpath, 'a') as w:
        line = '|'.join([lg, ly, l0, l, ls, ls0]) + '\n'
        w.write(line)
    return sol, obs

def get_cp_pairs():
    pass

def full_predict(a, b, years, dt):
    RE = RouteEvol(a, b, years, dt)
    obs0 = RE.get_net_cp_statistics()
    RE.rewire_net()
    obs = RE.get_net_cp_statistics()
    rew_steps = RE.i
    RF = rdf.RewireFit(RE.x, RE.n, obs0['na'])
    sol = RF.solve_randx0(5)
    return obs0, obs, rew_steps, sol

def cp_correlation_ideal(laa, lbb, lab, Na, Nb, a):
    """
    This function is wrong, attempts to measure observed values
    """
    N = Na + Nb
    L = laa + lbb + lab
    mx = L / (N*(N-1))
    my = (Na*(Na-1) + 2*Na*Nb*a) / (N*(N-1))
    sx = np.sqrt(2*mx - 2*mx**2)
    sy = np.sqrt(2*my - 2*my**2)

    t1 = laa - my*laa + mx*Na*(Na-1)*(my - 1)
    t2 = a*lab - my*lab - mx*Na*Nb + Nb*Na*my*mx
    t3 = Nb*(Nb-1)*mx*my - my*lbb

    res = (t1 + t2 + t3) / (sx * sy * N)
    return res

hub_ids = ['10397', '11298', '11292', '13930', '12892', '11057', '12889',
       '14107', '13204', '14747', '13303', '12266', '12478', '11697',
       '11618', '14771', '13487', '11433', '10721', '14869', '14100',
       '10821', '15304', '14679', '13232', '12953', '10693', '12264',
       '11259', '11278', '14057', '16440', '12191', '12173', '15016',
       '14635', '14893', '13495', '14843', '14492', '14831', '13796',
       '13198', '11042', '12339', '14683', '14908', '14122', '11193',
       '11066', '14027', '12451', '13342', '13891', '10299', '10529',
       '13830', '10800', '13871', '13244', '10713', '14570', '10994',
       '13851']

#hub_codes = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'LAS', 'PHX', 'MCO',
#       'SEA', 'MIA', 'IAH', 'JFK', 'FLL', 'EWR', 'SFO', 'MSP', 'DTW',
#       'BOS', 'SLC', 'PHL', 'BWI', 'TPA', 'SAN', 'MDW', 'LGA', 'BNA',
#       'IAD', 'DAL', 'DCA', 'PDX', 'AUS', 'HOU', 'HNL', 'STL', 'RSW',
#       'SMF', 'MSY', 'SJU', 'RDU', 'SJC', 'OAK', 'MCI', 'CLE', 'IND',
#       'SAT', 'SNA', 'PIT', 'CVG', 'CMH', 'PBI', 'JAX', 'MKE', 'ONT',
#       'ANC', 'BDL', 'OGG', 'BUR', 'OMA', 'MEM', 'BOI', 'RNO', 'CHS',
#       'OKC']

if __name__ == '__main__':
    import argparse

    outpath='airport/results_pairs.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--a', nargs='+', type=int, default=[1, 9])
    parser.add_argument('--b', nargs='+', type=int, default=list(range(2, 9)))
    parser.add_argument('--years', nargs='+', type=int, default=[1993, 2000])
    parser.add_argument('--dt', type=int, default=4)
    args = parser.parse_args()

    full_run(args.a, args.b, args.years[:2], args.dt, outpath)
