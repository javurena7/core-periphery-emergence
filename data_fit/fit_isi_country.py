import numpy as np
import networkx as nx
from collections import Counter
import pandas as pd
import copy
import pickle
import os

citation_path = '../data/isi/All_19002013_CitationList_Mapped_samp.txt' ##SAMPLE
##fields_path = '../data/isi/major_fields_total.txt'
metadata = '../data/isi/All_19002013_BasicInformation_Sample.txt'

citation_path = '/m/cs/scratch/isi/javier/All_19002013_CitationList_Mapped.txt'
#fields_path = '../major_fields_total.txt'
metadata = '/m/cs/scratch/networks/data/isi/WoS-Derived/All_19002013_BasicInformation.txt'

countries = 'isi/country_counts.txt'
outpath = 'cp_country_results.txt'

def get_country_groups():
    grps = pd.read_csv(countries, sep='|', names=['ctry', 'cnt', 'grp'])
    grps_dict = dict(grps[['ctry', 'grp']].values)
    grps['ctryr'] = grps['ctry'].apply(lambda x: x.replace(' ', '')) #FOR COUNTRY-LEVEL
    grps_dict = dict(grps[['ctry', 'ctryr']].values) #TODO: REMOVE IF DOING REGIONS
    return grps_dict

grps_dict = get_country_groups()

    #METADATA
    #df = pd.read_csv(metadata, sep='|', names=['ID', 'Number', 'lang', 'doctype', 'year', 'date' , 'nauthors', 'naddress','npags', 'nref', 'nfunding', 'ngrantnum', 'authors', 'country', 'city', 'pc', 'state', 'street', 'org'], dtype=str)

def categorize_countries(df):
    df['country'] = df.country.apply(_remove_repeated)
    df['area'] = df.country.apply(_areas)
    return df

def _get_area(x):
    return grps_dict.get(x, None)

def _areas(x):
    if '%' in x:
        a = set([_get_area(r) for r in x.split('%')])
        return a
    else:
        return set([_get_area(x)])


def _remove_repeated(x):
    x = set(x.lower().split('%'))
    if len(x) == 1:
        return x.pop()
    else:
        return '%'.join(list(x))


class ISIData(object):
    """
    Class for obtaining network evolution data for citation networks, where the groups are determined by countries in periods between years
    """
    def __init__(self, f1, f2, years=(0, np.inf), inits={}):
        self.f1 = f1
        self.f2 = f2

        self.x = {}
        self.n = {}
        self.i = 0
        self.get_metadata(years[0], years[1])
        self.nnodes = {'a': set(), 'b': set()}
        if not inits:
            #self.net = nx.empty_graph()
            self.deg = {}
            self.deg_dist = {'a': {}, 'b': {}}
            self.tots = {'laa': 0,
                    'lbb': 0,
                    'lab': 0,
                    'lba': 0}
        else:
            #self.net = net0
            self.deg = inits['deg']
            self.deg_dist_n0()
            self.tots = inits['tots']
        self.Na_0 = len(set(self.deg).intersection(self.groups['a']))
        self.N_0 = len(self.deg)

        self.years = years

    def get_metadata(self, min_yr=0, top_yr=np.inf):
        """
        Read metadata line by line, and categorize countries
        """
        groups = {'a': set(), 'b': set()}
        cit_papers = set()
        n_idx = 1 #Index of "Number" (paper id)
        y_idx = 4 #Index of year
        ctry_idx = 13 #Index for country
        with open(metadata, 'r') as r:
            line = r.readline()
            while line:
                line = line.split('|')
                yr = int(line[y_idx])
                ppr_n = line[n_idx]
                checked = ppr_n in groups['a'] or ppr_n in groups['b']
                if (yr < top_yr) and (ppr_n != 'None') and (not checked): #Check that we are not over_years
                    ctry = _remove_repeated(line[ctry_idx])
                    areas = _areas(ctry)
                    if areas.intersection(self.f1):
                        if not areas.intersection(self.f2):
                            groups['a'].add(ppr_n)
                            if yr >= min_yr:
                                cit_papers.add(ppr_n)
                    elif areas.intersection(self.f2):
                        groups['b'].add(ppr_n)
                        if yr >= min_yr:
                            cit_papers.add(ppr_n)
                line = r.readline()
        self.groups = groups
        self.cit_papers = cit_papers

    #def get_groups(self, years):
    #    groups = {}
    #    df = self.get_metadata(years[1])
    #    df = df[['ID', 'Number', 'country', 'year']]
    #    df['year'] = df.year.astype(int)
    #    df = df[(df.year > years[0]) & (df.year < years[1])]
    #    df = categorize_countries(df)


    #   groups['a'] = set(df.Number[df.area.isin(self.f1)].values)
    #    groups['b'] = set(df.Number[df.area.isin(self.f2)].values)
    #    self.groups = groups

    def grow_net(self):
        with open(citation_path, 'r') as r:
            line = r.readline()
            while line:
                self.add_edges(line)
                line = r.readline()

    def add_edges(self, line):
        line = line.replace('\n', '')
        try:
            new, cits = line.split('|')
        except:
            new, cits = '', ''
        new_g = ''
        #cit_papers contains the set of papers within selected years
        if new in self.cit_papers:
            if self.check_a(new):
                new_g = 'a'
            elif self.check_b(new):
                new_g = 'b'

        if new_g:
            cits = cits.split(',')
            add = False
            self.current_x = []
            self.update_net_stats()
            cits = set([cit for cit in cits if cit != new])
            targets = set()
            for cit in cits:
                if self.check_a(cit):
                    #self.net.add_edge(new, cit)
                    self.tots['l'+new_g+'a'] += 1
                    self.nnodes['a'].add(cit)
                    self.update_add_stats(new_g, 'a', cit)
                    targets.add(cit)
                elif self.check_b(cit):
                    #self.net.add_edge(new, cit)
                    self.tots['l'+new_g+'b'] += 1
                    self.nnodes['b'].add(cit)
                    self.update_add_stats(new_g, 'b', cit)
                    targets.add(cit)

            if targets:
                self.nnodes[new_g].add(new)
                for cit in targets:
                    self.deg[cit] = self.deg.get(cit, 0) + 1
                counts = Counter(self.current_x)
                fx = [[x[0], x[1], x[2], k] for x, k in counts.items()]
                self.deg[new] = self.deg.get(new, 0) + len(self.current_x)
                self.x[self.i] = fx
                nx = self.current_n
                self.n[self.i] = nx
                self.update_deg_dist(fx)
                self.i += 1


    def deg_dist_n0(self):
        a_nodes = []
        b_nodes = []
        for node, deg in self.deg.items():
            #deg = self.net.degree(node)
            if self.check_a(node):
                a_nodes.append(deg)
            else:
                b_nodes.append(deg)
        a_deg = Counter(a_nodes)
        b_deg = Counter(b_nodes)
        self.deg_dist = {'a': a_deg, 'b': b_deg}


    def update_net_stats(self):

        pa = sum([k*x for k, x in self.deg_dist['a'].items()])
        pb = sum([k*x for k, x in self.deg_dist['b'].items()])
        ua = sum(self.deg_dist['a'].values())
        ub = sum(self.deg_dist['b'].values())
        self.current_n = [pa, pb, ua, ub]

    def update_deg_dist(self, x_i): #x_i is fx
    # Light version
        n_i = copy.deepcopy(self.deg_dist)
        for x in x_i:
            deg = x[1] #target degree
            nk = x[2] #num of nodes of degree tgt_deg in tgt group
            cnt = x[3] # number of chosen nodes of degree tgt_deg in tgt group
            tgt = x[0][1]
            src = x[0][0]
            n_i[tgt][deg] = max([n_i[tgt].get(deg, 0) - cnt, 0])
            n_i[tgt][deg+1] = n_i[tgt].get(deg+1, 0) + cnt

        tot_deg = sum([x[3] for x in x_i]) #total links added to new link
        if tot_deg:
            n_i[src][tot_deg] = n_i[src].get(tot_deg, 0) + 1
        clean_ni = {'a': {}, 'b': {}}
        for sg in clean_ni:
            clean_ni[sg] = {k: v for k, v in n_i[sg].items() if v > 0}

        #CHECK that the group-level dists are keps constant
        #tota = sum([k*v for k, v in clean_ni['a'].items()])
        #totb = sum([k*v for k, v in clean_ni['b'].items()])
        #tot = sum(self.deg.values())
        #if tot != (tota + totb):
        #    import pdb; pdb.set_trace()
        self.deg_dist = clean_ni

    def update_add_stats(self, sgroup, tgroup, tgt):
        """
        xg - link type
        tgt_deg - degree of target node
        nk - number of links of degree tgt_deg in the target group
        """
        xg = sgroup + tgroup
        tgt_deg = self.deg.get(tgt, 0) #self.net.degree(tgt)
        nk = self.deg_dist[tgroup].get(tgt_deg, 1)
        self.current_x.append((xg, tgt_deg, nk))


    def check_a(self, x):
        if x in self.groups['a']:
            return True
        else:
            return False

    def check_b(self, x):
        if x in self.groups['b']:
            return True
        else:
            return False

    def get_group_numbers(self):
        """
        Gets group stats - nums of nodes added and in total
        """
        #TOTAL
        Na_tot = len(set(self.deg).intersection(self.groups['a']))
        N_tot = len(self.deg)

        #Added
        Na_add = Na_tot - self.Na_0
        N_add = N_tot - self.N_0

        na_tot = Na_tot / N_tot
        na_add = Na_add / N_add
        self.nas = {'na_tot': na_tot, 'na_add': na_add,
                'N_tot': N_tot, 'N_add': N_add}


    def print_tots(self, totspath=''):
        res = self.tots
        resline = '{}|{}|{}|'.format(res['laa'], res['lbb'] , res['lab'] + res['lba'])
        nnodes = '{}|{}|'.format(len(self.nnodes['a']), len(self.nnodes['b']))
        yy = '{}|{}'.format(self.years[0], self.years[1])
        line = '{}|{}|'.format(self.f1, self.f2) + resline + nnodes + yy
        if totspath:
            #totspath = outpath
            with open(totspath, 'a') as w:
                w.write(line+'\n')
        else:
            return line

    def get_obs(self):
        """
        Get network observations - pmat, densities, group sizes, onion measures
        """
        Ls = self.tots
        L = sum(Ls.values())
        paa = Ls['laa'] / L ; pbb = Ls['lbb'] / L ; pab = 1-paa-pbb
        self.get_group_numbers(); N = self.nas['N_tot']; na = self.nas['na_tot']

        rho = 2 * L / (N * (N-1))
        obs = {'paa': paa, 'pbb': pbb, 'pab': pab, 'na': na,
                'N': N, 'L': L, 'rho': rho}

        if (na > 0) and (na < 1):
            rhoa = paa * rho / (na**2)
            rhob = pbb * rho / ((1-na)**2)
            rhoab = pab * rho / (2*na*(1-na))

            if rhoa > max([rhob, rhoab]):
                core = 'a'
                cp_meas = (rhoab - rhob) / (rhoab + rhob)
            elif rhob > max([rhoa, rhoab]):
                core = 'b'
                cp_meas = (rhoab - rhoa) / (rhoab + rhoa)
            else:
                core = ''
                cp_meas = np.nan
        else:
            rhoa, rhoab, rhob, core, cp_meas = np.nan, np.nan, np.nan, np.nan, np.nan
        obs.update({'rhoa': rhoa, 'rhoab': rhoab, 'rhob': rhob,
                    'core': core, 'cp_meas': cp_meas})
        return obs


    def get_data(self):
        self.grow_net()
        x, n = {}, {}
        for (i, xi), (j, ni) in zip(self.x.items(), self.n.items()):
            xj = [s for s in xi if s[1] > 0]
            if xj:
                x[i] = xj
                n[i] = ni
        if x:
            obs = self.get_obs()
        else:
            obs = {}
        return x, n, obs


def get_all_CP_measures(f1, f2, fpath='isi/onion'):
    """
    Compute data for CP onion measures during a range of predetermined periods
    Initial conditions are cumulative - end network includes data from all periods
    """
    #areas = ['USA', 'WEST', 'ASIA', 'EBLO', 'LATIN', 'AFRICA']
    years = [1900, 1940, 1960, 1970, 1980] + list(range(1985, 2001, 5))
    yrs0 = ''
    csvpath = f'{fpath}/full_onion_measures.csv'
    names = [f1, f2]; ef1 = f1.split('-'); ef2 = f2.split('-')
    cols = ['a', 'b', 'yrs', 'yrs0', 'paa', 'pbb', 'pab', 'na', 'N', 'L',
            'rho', 'rhoa', 'rhob', 'rhoab', 'core', 'cp_meas']

    def _get_inits(yrs0):
        """
        Simple function for reading initial conditions if they exit
        """
        if yrs0 != '':
            initpath = '{}/{}_{}.p'.format(fpath, *names)
            data = read_picklepath(initpath)
            datay = data.get(yrs0, {})
            inits = {}
            if datay:
                inits = {'deg': copy.deepcopy(datay['deg']),
                        'tots': copy.deepcopy(datay['tots']),
                        'yrs0': yrs0}
            del data
            return inits
        return {}

    for yrs in zip(years[:-1], years[1:]):
        yrsn = '{}-{}'.format(*yrs)
        picklepath = '{}/{}_{}.p'.format(fpath, *names)
        if not years_in_pickle(yrsn, picklepath): #POSSIBLY VERY INEFFICIENT
            print('{} - {}: {}'.format(*names, yrsn))
            # GET DATA
            inits = _get_inits(yrs0)
            ID = ISIData(ef1, ef2, yrs, inits)
            x, n, obs = ID.get_data()

            # SAVE CSV DATA
            obs['a'] = names[0]; obs['b'] = names[1]
            obs['yrs'] = yrsn; obs['yrs0'] = inits.get('yrs0', '')
            #if os.path.exists(csvpath):
            #    df = pd.read_csv(csvpath)
            #else:
            #    df = pd.DataFrame(columns=cols)

            #df = df.append(obs, ignore_index=True)
            #df.to_csv(csvpath, index=None)

            # SAVE PICKLE DATA
            data = read_picklepath(picklepath)
            data[yrsn] = {'x': x, 'n': n, 'obs': obs, 'deg': ID.deg, 'tots': ID.tots}
            with open(picklepath, 'wb') as w:
                pickle.dump(data, w)
            del data; del ID
        yrs0 = yrsn

def read_picklepath(path):
    if os.path.exists(path):
        with open(path, 'rb') as r:
            data = pickle.load(r)
        return data
    else:
        return {}

def years_in_pickle(yrs, path):
    data = read_picklepath(path)
    if yrs in data:
        del data
        return True
    else:
        del data
        return False

def fit_data(f1, f2, fpath='isi/onion_minor_region', outpath='isi/onion_fit', pp='', fit_c=False):
    import growth_degree_fit as gdf
    names = [f1, f2]
    picklepath = '{}/{}_{}.p'.format(fpath, *names) if not pp else pp
    data = read_picklepath(picklepath)
    outname = '{}/{}_{}.p'.format(outpath, *names)
    fdata = {}
    for yrs in data:
        datay = data.get(yrs, {})
        obs = datay['obs']

        #### TODO: CHECK THIS IS CORRECT
        if datay['x']:
            rhoab = 2*obs['rhoab']; rhob = obs['rhob']; rhoa = obs['rhoa']
            if rhoa > max([rhoab, rhob]):
                cm = (rhoab - rhob) / (rhoab + rhob); obs['core'] = 'a'
            elif rhob > max([rhoab, rhoa]):
                cm = (rhoab - rhoa) / (rhoab + rhoa); obs['core'] = 'b'
            else:
                cm = np.nan; obs['core'] = ''
            obs['cp_meas'] = cm

            GF = gdf.GrowthFit(datay['x'], datay['n'], obs['na'])
            sol = GF.solve(); obs['c'] = sol[2]
            obs['sa'] = sol[0]; obs['sb'] = sol[1]
            sol0 = GF.solve_c0()
            obs['sa0'] = sol0[0]; obs['sb0'] = sol0[1]

            #LLik ratios
            llik_c = GF.loglik(sol)
            llik_c0 = GF.loglik0(sol0)
            t = 2*(llik_c - llik_c0)
            obs['llik_c'] = llik_c
            obs['llik_c0'] = llik_c0
            obs['chi'] = t

            print(f'Done {yrs} : {cm}')
            print(sol)
            fdata[yrs] = obs
            pickle.dump(fdata, open(outname, 'wb'))


if __name__ == '__main__':
        import sys
        from os.path import exists
        f1, f2 = sys.argv[1:3]
        get_all_CP_measures(f1, f2, fpath='isi/onion_minor_region')
        fit_data(f1, f2, fpath='isi/onion_minor_region', outpath='isi/onion_fit')
        #TODO: ADAPT CODE FOR FITTING DATA OF REGIONS THAT HAVE CP
        #   func: fit_data with fit_c=False as well
        # TODO: add smaller regions (between-country relationships, etc)

        """
        #NOTE: f1 and f2 must be lists, python fit_isi_country.py '["USA"]' '["WEST"]'
        import growth_degree_fit as gdf

        f1, f2 = sys.argv[1:3]
        yrs = sys.argv[3:5]
        yr_0 = eval(yrs[0])
        yr_range = eval(yrs[1])
        net0 = nx.empty_graph()
        tots0 = {'laa': 0,
                'lbb': 0,
                'lab': 0,
                'lba': 0}
        for yr in range(yr_0, 2011, yr_range):
            yrs = (yr, yr + yr_range)
            ID = ISIData(eval(f1), eval(f2), yrs, net0=net0, tots0=tots0)
            x, n, na = ID.get_data()

            f1c, f2c = '-'.join(eval(f1)), '-'.join(eval(f2))
            fcountries = 'A%{}_B%{}'.format(f1c, f2c)
            fyrs = str(yrs[0]) + str(yrs[1])
            #cp_name = 'isi/country_estimated_params_{}.txt'.format(fcountries)
            es_name = 'isi/country_evol_incn0_{}.txt'.format(fcountries)

            cp_line = ID.print_tots()
            GF = gdf.GrowthFit(x, n, na)
            sol = GF.solve()
            #f1|f2|sa|sb|c|na
            #totline: f1|f2|laa|lbb|lab|Na|Nb|
            line = cp_line + '|{}|{}|{}|{}\n'.format(sol[0], sol[1], sol[2], na)
            print(line)
            if not exists(es_name):
                with open(es_name, 'a') as w:
                    w.write('f1|f2|laa|lbb|lab|Na|Nb|y0|yn|sa|sb|c|na\n')
            with open(es_name, 'a') as w:
                w.write(line)

            sol0 = GF.solve_c0()
            line = cp_line + '|{}|{}|{}|{}\n'.format(sol0[0], sol0[1], 0, na)
            with open(es_name, 'a') as w:
                w.write(line)

            net0 = ID.net.copy()
            tots0 = ID.tots.copy()
            """


