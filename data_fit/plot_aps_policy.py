import plot_policies as pp
import sys; import pickle; import os
sys.path.append('..')
import matplotlib.pyplot as plt; plt.ion()
import get_fixed_points as gfp
from utils import pacs_utils as pu
import growth_degree_fit as gdf
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from itertools import product

group_names = json.load(open('utils/pacs_ref.json', 'r'))

#TODO: finnish plotting policies: get t_size and get policies


def plot_policies(a='', b=''):

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    prm_list = ['c', 'na', 'sa', 'sb']
    path = 'aps/fitted/'

    for (i, j) in product(range(4), range(4)):
        axs[i][j].spines['right'].set_visible(False)
        axs[i][j].spines['top'].set_visible(False)

    obs = get_data(a, b, path)
    for i, obs_i in obs.items():
        print(i)
        P0 = (1/3, 1/3)
        P = (obs_i['paa'], obs_i['pbb'])
        fx_params = {x: obs_i[x] for x in prm_list}
        for j, x in enumerate(prm_list):
            params = deepcopy(fx_params)
            pol_params = deepcopy(fx_params)
            obs_n = {'paa': P[0], 'pbb': P[1]}; rho = obs_i['rho']
            obs_0 = {'paa': P0[0], 'pbb': P0[1], x: params.pop(x)}
            #pol_steps = get_policies()
            legend = True if (i == 3) and (j == 0) else False
            pp.plot_fullrange_policy_grow(x, params, rho=rho, obs_0=obs_0, obs_n=obs_n, fig=fig, ax=axs[j, i], legend=legend)
            fig.savefig(path + f'{a}_{b}.pdf')
            P0 = P
    fig.suptitle('Growth fit, A: {} \nB: {}'.format(group_names[a], group_names[b]))
    fig.tight_layout()
    fig.savefig(path + f'{a}_{b}.pdf')


def get_data(a='', b='', ppath='aps/onion_fit/'):
    a = str(int(a)).zfill(2)
    b = str(int(b)).zfill(2)
    print('{}-{}'.format(a, b))
    path = ppath + '{}_{}.p'.format(a, b)
    if os.path.exists(path):
        obs = pickle.load(open(path, 'rb'))
    else:
        obs = _calculate_data(a, b, path) #, df, dfpath)
    return obs


def _calculate_data(a='', b='', path=''):
    from plot_boards import rho_stats
    #Note, if df is not None, path can't be empty
    x, n, obs = pu.cp_stats([a], [b])
    istart = 0
    for i, obs_i in obs.items():
        ra = gfp.rmax(obs_i['paa'], obs_i['pbb'], obs_i['na'], obs_i['rho'])
        rb = gfp.rmax(obs_i['pbb'], obs_i['paa'], 1-obs_i['na'], obs_i['rho'])
        obs_i['ra'] = ra; obs_i['rb'] = rb
        iend = obs_i['i']
        x_i = {j: v for j, v in x.items() if (j >= istart) & (j <= iend)}
        n_i = {j: v for j, v in n.items() if (j >= istart) & (j <= iend)}
        GF = gdf.GrowthFit(x_i, n_i, obs_i['na'])
        rho_stats(obs_i)
        print(obs_i)
        sa, sb, c = GF.solve()
        obs_i['c'] = c
        obs_i['sa'] = sa; obs_i['sb'] = sb
        sa0, sb0 = GF.solve_c0()
        obs_i['sa0'] = sa0; obs_i['sb0'] = sb0

        llik_c = GF.loglik((sa, sb, c))
        llik_c0 = GF.loglik0((sa0, sb0))

        t = 2*(llik_c - llik_c0)
        obs_i['llik_c'] = llik_c
        obs_i['llik_c0'] = llik_c0
        obs_i['chi'] = t

    obs['a'] = a; obs['b'] = b
    if path:
        pickle.dump(obs, open(path, 'wb'))
    return obs


def data_from_list(listpath='aps/all_onion_pairs.csv', outpath='aps/onion_fit/', lower_bound=-.2):
    """
    Main func for fitting that that has CP
    """
    df = pd.read_csv(listpath, sep='|')
    for i, row in df.iterrows():
        na = row.na
        if (row.cp_meas > lower_bound) and (min([na, 1-na]) > .05):
            if row.na < .5:
                _ = get_data(row.a, row.b, ppath=outpath)
            else:
                _ = get_data(row.b, row.a, ppath=outpath)


def fit_data_nocp(listpath='aps/all_onion_pairs.csv', outpath='aps/nocp/', na_bound=.3, pab_bound=.1):
    """
    Main func for fitting that that has CP
    """
    df = pd.read_csv(listpath, sep='|')
    for i, row in df.iterrows():
        na = row.na
        if (row.pab >= pab_bound) and (min([na, 1-na]) > na_bound):
            if row.na < .5:
                _ = get_data(row.a, row.b, ppath=outpath)
            else:
                _ = get_data(row.b, row.a, ppath=outpath)

if __name__ == '__main__':
    data_from_list(outpath='aps/onion_fit/') #FOR FITTING CP
    #fit_data_nocp(na_bound=.3, pab_bound=.1)
