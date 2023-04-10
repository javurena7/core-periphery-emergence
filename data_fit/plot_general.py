import pickle as p
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
from matplotlib import cm
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../ps_plots')
from plot_fixed_points import plot_mini_net
import cp_measure as cpm
import os
import networkx as nx
import get_fixed_points as gfp
from itertools import product

#isi/country_evol works a bit
def read_isi_country(path='isi/country_evol/'):
    paths = ['isi/country_evol/', 'isi/short_range/']
    logs = []
    for path in paths:
        logs += [path + x for x in os.listdir(path)]
    isi_data = []
    for log in logs:
        data = {}

        df = pd.read_csv(log, sep='|')
        df['L'] = df.laa + df.lab + df.lbb
        df['N'] = df.Na + df.Nb
        df = df[(df.L > 0) & (df.N > 1) & (df.c > 0)]
        #import pdb; pdb.set_trace()
        #dc = df.iloc[-1, :]
        for _, dc in df.iterrows():
            L = dc.L #laa + dc.lbb + dc.lab
            N = dc.N #Na + dc.Nb
            data['set'] = 'isi'
            data['name'] = eval(dc.f1)[0] + '.' +  eval(dc.f2)[0] + f'.{dc.y0}-{dc.yn}'
            data['paa'] = dc.laa / L; data['pbb'] = dc.lbb / L
            data['pab'] = 1 - data['paa'] - data['pbb']
            data['na'] = dc.Na / N; data['c'] = dc.c
            data['sa'] = dc.sa; data['sb'] = dc.sb
            rho = 2*L / (N*(N-1))
            data['rho'] = rho
            na = data['na']

            data['N'] = N
            data['L'] = L

            rhoa = data['paa'] * rho / (na**2) if na > 0 else 0
            rhob = data['pbb'] * rho / ((1-na)**2) if 1-na>0 else 0
            rhoab = data['pab'] * rho / (na*(1-na)) if na*(1-na)>0 else 0
            data['rhoa'] = rhoa; data['rhob'] = rhob; data['rhoab'] = rhoab
            if rhoa > max([rhoab, rhob]):
                data['core'] = 'a'
                data['cp_meas'] = (rhoab-rhob)/(rhoab+rhob) if rhoab+rhob>0 else 0
            elif rhob > max([rhoab, rhoa]):
                data['core'] = 'b'
                data['cp_meas'] = (rhoab-rhoa)/(rhoab+rhoa) if rhoab+rhoa>0 else 0
        #try: ### TODO: fails in a simple bug, overall we need to remove this
        #    dc = df.iloc[-2, :]
        #    L = dc.laa + dc.lbb + dc.lab
        #    N = dc.Na + dc.Nb
        #    data['paa0'] = dc.laa / L; data['pbb0'] = dc.lbb / L
        #    data['pab0'] = 1 - data['paa0'] - data['pbb0']
        #except:
        #    import pdb; pdb.set_trace()
        if (data['cp_meas'] > -.1) & (data['na'] > .01) & (data['na']<.99) & (data['c'] > .08):
            isi_data.append(data)
    return isi_data

def read_isi(path='isi/onion_fit/'):
    logs = [path + x for x in os.listdir(path) if x.endswith('.p')]
    names = [x[:-2] for x in os.listdir(path) if x.endswith('.p')]
    isi_data = []
    for log, name in zip(logs, names):
        dc = p.load(open(log, 'rb'))
        for yrs, data in dc.items():
            data['set'] = 'isi'
            data['name'] = name + ': ' + yrs
            if (data['cp_meas'] > -.1) & (data['na'] > .05) & (data['na'] < .95):
                data['rhoab'] = 2*data['rhoab']
                isi_data.append(check_minority_group(data))
    return isi_data

def check_minority_group(data):
    """
    Check if minority group is group a
    """
    if data['na'] <= .5:
        return data
    else:
        new = {}
        for k, v in data.items():
            new[_ch_gr(k)] = v
        new['na'] = 1-new['nb']
        new['core'] = _ch_gr(data['core'])
        new['name'] = f"{new['a']}_{new['b']}: {new['yrs']}"
        new['cp_meas'] = new['cp_mebs'] #lazy coding
    return new


def _ch_gr(k):
    if ('a' in k) and not ('b' in k):
        return k.replace('a', 'b')
    elif ('b' in k) and not ('a' in k):
        return k.replace('b', 'a')
    else:
        return k


def read_aps(path='aps/onion_fit/'):
    logs = [path + x for x in os.listdir(path) if x.endswith('.p')]
    names = [x[:-2] for x in os.listdir(path) if x.endswith('.p')]
    aps_data = []
    for log, name in zip(logs, names):
        dc = p.load(open(log, 'rb'))
        for i, data in dc.items():
            if isinstance(i, int):
                data['set'] = 'aps'
                data['name'] = str(i)
                data['a'] = name.split('_')[0]
                data['b'] = name.split('_')[1]
                data['pab'] = 1 - data['paa'] - data['pbb']
                if data['cp_meas'] > -.15:
                    aps_data.append(data)
    return aps_data

def read_twitter(path='twitter/onion_fit/', lwr=0.1):
    twt_data = []
    logs = [path + x for x in os.listdir(path) if x.startswith('opt')]
    for log in logs:
        dlog = p.load(open(log, 'rb'))
        for i, dc in dlog.items():
            data = {}
            data['set'] = 'twitter'
            data['name'] = i
            data['a'] = 'pol'
            data['b'] = 'npol'
            data.update(dc['obs'])
            data.update(dc['fit'])
            data.update({k.replace('r', 'rho').replace('aa', 'a').replace('bb', 'b'): v for k, v in dc['rho'].items()})
            data['DT'] = dc['DT']
            data['dates'] = dc['dates']
            data['ttype'] = dc['ttype']

            rho = 2*data['L'] / (data['N']**2); na = data['na']
            rhoab = (data['pab']) * rho / (2*na*(1-na))
            data['rhoab'] = rhoab; rhoa = data['rhoa']; rhob = data['rhob']
            if rhoa > max([rhoab, rhob]) and (rhoab - rhob)/(rhoab+rhob) > lwr:
                data['cp_meas'] =  (rhoab - rhob)/(rhoab+rhob)
                data['core'] =  'a'
                twt_data.append(data)
            elif rhob > max([rhoab, rhoa]) and (rhoab-rhoa)/(rhoab+rhoa) > lwr:
                data['cp_meas'] =  (rhoab - rhoa)/(rhoab+rhoa)
                data['core'] =  'b'
                twt_data.append(data)
    return twt_data


def read_airport(path='airport/onion_fit/', lwr=.1):
    apt_data = []
    logs = [path + x for x in os.listdir(path)]
    for log in logs:
        dc = p.load(open(log, 'rb'))
        data = {}
        data['set'] = 'airport'
        name = log.replace('path', '')

        data['name'] = name + '_{}-{}'.format(int(dc['y0']), int(dc['yn']))
        data['years'] = [int(dc['y0']), int(dc['yn'])]
        var = ['a', 'b', 'paa', 'pbb', 'rhoa', 'rhob', 'rhoab', 'llik_c', 'llik_c0', 'chi', 'N', 'L', 'c', 'sa', 'sb', 'sa0', 'sb0', 'na', 'dt', 'rho']
        for k in var:
            data[k] = dc[k]
        data['pab'] = 1 - data['paa'] - data['pbb']
        rhoa = data['rhoa']
        rhob = data['rhob']
        rhoab = data['rhoab']

        if rhoa > max([rhoab, rhob]) and (rhoab - rhob)/(rhoab+rhob) > lwr:
            data['cp_meas'] =  (rhoab - rhob)/(rhoab+rhob)
            data['core'] =  'a'
            apt_data.append(data)
        elif rhob > max([rhoab, rhoa]) and (rhoab-rhoa)/(rhoab+rhoa) > lwr:
            data['cp_meas'] =  (rhoab - rhoa)/(rhoab+rhoa)
            data['core'] =  'b'
            apt_data.append(data)
    return apt_data



def read_boards():
    dat = p.load(open('boards/onion_fit.p', 'rb'))
    brd_data = []
    for i, dc in dat.items():
        data = {}
        data['set'] = 'board'
        data['name'] = i
        data['a'] = 'w'; data['b'] = 'm'
        data.update(dc)
        brd_data.append(data)
    return brd_data

def read_airport_homophilous_pairs(path='airport/results_pairs.txt'):
    #### homophilous pairs are to check behaviour when
    #### both datasets are estimated to be homophilous
    df = pd.read_csv(path, sep='|')
    df = df[(df.sa > .55) & (df.sb > .55)]
    df['Na'] = df.N * df.na
    df['paa'] = df.laa / df.L
    df['pbb'] = df.lbb / df.L
    df['pab'] = df.lab / df.L
    df['rhoa'] = 2*df.laa / (df.Na**2)
    df['rhob'] = 2*df.lbb / ((df.N-df.Na)**2)
    df['rhoab'] = df.lab / ((df.N-df.Na)*df.Na)
    apt_hdata = list(dict(v) for k, v in df.iterrows())
    for x in apt_hdata:
        rhoa = x['rhoa']
        rhob = x['rhob']
        rhoab = x['rhoab']
        if rhoa > max([rhob, rhoab]):
            x['cp_meas'] = (rhoab-rhob)/(rhoab+rhob)
        if rhob > max([rhoa, rhoab]):
            x['cp_meas'] = (rhoab-rhoa)/(rhoab+rhoa)
    return apt_hdata

def read_aps_homophilous_pairs(path='aps/nocp/'):
    #### homophilous pairs are to check behaviour when
    #### both datasets are estimated to be homophilous
    logs = [path + x for x in os.listdir(path) if x.endswith('.p')]
    names = [x[:-2] for x in os.listdir(path) if x.endswith('.p')]
    aps_hdata = []
    for log, name in zip(logs, names):
        dc = p.load(open(log, 'rb'))
        for i, data in dc.items():
            if isinstance(i, int):
                data['set'] = 'aps'
                data['name'] = name + '_' + str(i)
                data['pab'] = 1 - data['paa'] - data['pbb']
                aps_hdata.append(data)
                #rho = data['rho']; na = data['na']
                #if (na > 0) and (na < 1):
    return aps_hdata

def get_all_sets():
    sets_data = {}
    for dset in ['airport', 'twitter', 'isi', 'aps', 'boards']:
        fdata = eval(f'read_{dset}()')
        sets_data[dset] = fdata
    return sets_data

def plot_board_evolution():
    fig, ax = plt.subplots(figsize=(3, 3))
    data = read_boards()
    c = [d['c'] for d in data]
    sa = [d['sa'] for d in data]
    sb = [d['sb'] for d in data]
    na = [d['na'] for d in data]
    vnams = ['sa', 'sb', 'c', 'na']
    xt = ['2002-\n2003', '2003-\n2005', '2005-\n2007', '2007-\n2009', '2009-\n2011']
    cmap = cm.get_cmap('Set1')
    col = {x: cmap(i/10+.05) for i, x in enumerate(vnams)}
    labs = {'c': r'$c$',
            'sa': r'$s_a$',
            'sb': r'$s_b$',
            'na': r'$n_a$'}
    for var in vnams:
        ax.plot(eval(var), '-', color=col[var], alpha=.7)
        ax.plot(eval(var), '.', color=col[var], label=labs[var])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(xt)
    ax.axhline(.4, ls='--', color='grey', alpha=.3)
    ax.legend(loc=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('boards/fig5_board_evol.pdf')


def fit_all_sets(overwrite=False, allpath='all_fitted_sets.p'):
    if os.path.exists(allpath) and (not overwrite):
        sdata = p.load(open(allpath, 'rb'))
        return sdata
    else:
        sdata = get_all_sets()
        rtypes = {'airport': 'r', 'twitter': 'r', 'boards': 'r', 'isi': 'g', 'aps': 'g'}
        for dset, data in sdata.items():
            rtype = rtypes[dset]
            dfixed_point(data, rtype)
        p.dump(sdata, open(allpath, 'wb'))
        return sdata

def fit_homophilous_sets(overwrite=False, path='nocp_fitted_sets.p'):
    if os.path.exists(path) and (not overwrite):
        sdata = p.load(open(path, 'rb'))
        return sdata
    else:
        sdata = {}; rtypes = {'airport': 'r', 'aps': 'g'}
        for dset in ['airport', 'aps']:
            fdata = eval(f'read_{dset}_homophilous_pairs()')
            rtype = rtypes[dset]
            dfixed_point(fdata, rtype)
            sdata[dset] = fdata
        p.dump(sdata, open(path, 'wb'))
        return sdata

def dfixed_point(data, rtype):
    """
    Find theoretical fixed points given fitted paramters
    """
    for dat in data:
        frhoc, frho0, frho1 = dat_to_fixp(dat, rtype)
        hpr = np.array([dens_to_cpmeas(f) for f in frhoc])
        try:
            cp_meas, core = hpr[:, 0], hpr[:, 1]
        except:
            cp_meas = np.nan; core = 'n'
        hpr0 = np.array([dens_to_cpmeas(f0) for f0 in frho0])
        cp_meas0, core0 = hpr0[:, 0], hpr0[:, 1]
        hpr1 = np.array([dens_to_cpmeas(f1) for f1 in frho1])
        cp_meas1, core1 = hpr1[:, 0], hpr1[:, 1]

        dat['f_rho'] = frhoc; dat['f_rho0'] = frho0; dat['f_rho1'] = frho1
        dat['f_cp_meas'] = cp_meas; dat['f_cp_meas0'] = cp_meas0;
        dat['f_cp_meas1'] = cp_meas1
        dat['f_core'] = core; dat['f_core0'] = core0; dat['f_core1'] = core1

def plot_all_c_effect_cp():
    sdata = fit_all_sets()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    colors = {'airport': 'r', 'twitter': 'g', 'boards': 'b', 'isi': 'k', 'aps': 'c'}
    rtypes = {'airport': 'r', 'twitter': 'r', 'boards': 'r', 'isi': 'g', 'aps': 'g'}
    line = np.linspace(0, 1, 100)
    axs[0].plot(line, line, color='grey', alpha=.4)
    axs[1].plot(line, line, color='grey', alpha=.4)
    for dset, data in sdata.items():
        rtype = rtypes[dset]
        if rtype == 'r':
            plot_c_effect_cp(data, axs[1], colors[dset])
        else:
            plot_c_effect_cp(data, axs[0], colors[dset])

def plot_c_effect_cp(data, ax, color=''):
    for dat in data:
        #frhoc, frho0 = dat['f_rho'], dat['f_rho1']
        na = dat['na']
        if np.abs(na-.5) < .45:
            cp, cp0 = dat['f_cp_meas'], dat['f_cp_meas1']
            cpe = dat['f_cp_meas0']
            c = dat['c']
            #amp = cp / (cp+cp0)
            ax.plot(c, cp, '.', color=color)
            #ax.plot(c, cpe, 'o', color=color)
            ax.plot(c, cp0, 'x', color=color)

def plot_p_comp_theo(n_samp, rtype='r', c_range=(.1, .9)):
    """
    Theretical effect on p-matrix based on sa, sb and c; from random sampling the phase space
    """
    fig, axs = plt.subplots(3, 5, figsize=(10.35, 3*2.8), sharex=True, sharey=True)
    c_ranges = [(.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1)]
    i = 0; tot = n_samp*5*3
    for col, c_range in enumerate(c_ranges):
        sas, sbs = [], []
        for _ in range(n_samp):
            if np.random.rand() < .5:
                sa = np.random.uniform(.05, .95)
                sb = np.random.uniform(.05, .95)
                #while np.abs(sa - sb) < .1:
                #    sb = np.random.uniform(.05, .95)
            else:
                sa = np.random.uniform(.05, .95)
                sb = np.random.uniform(.05, .95)
                #while np.abs(sa - sb) < .1:
                #    sa = np.random.uniform(.05, .95)
            sas.append(sa); sbs.append(sb)
        ns = [.1, .25, .5]
        for row, na in enumerate(ns):
            ax = axs[row, col]
            ax.axhline(0, color='grey', alpha=.3)
            cs = np.random.uniform(*c_range, n_samp)
            for sa, sb, c in zip(sas, sbs, cs):
                print(i/tot)
                if rtype=='r':
                    fp0 = gfp.rewire_fixed_points(c=.0001, na=na, sa=sa, sb=sb)[0]
                    fps = gfp.rewire_fixed_points(c=c, na=na, sa=sa, sb=sb)
                else:
                    fp0 = gfp.growth_fixed_points(c=.0001, na=na, sa=sa, sb=sb)[0]
                    fps = gfp.growth_fixed_points(c=c, na=na, sa=sa, sb=sb)

                p0 = np.array([fp0[0], 1-fp0[0]-fp0[1], fp0[1]])
                for fp in fps[::2]:
                    p = np.array([fp[0], 1-fp[0]-fp[1], fp[1]])
                    sv = (sa, .5*(sa+sb), sb)

                    c_stat = (sa-sb)/(2*(sa+sb)) + .5
                    color = cm.coolwarm(c_stat)
                    alpha = .5
                    ax.plot(sv, p-p0, color=color, alpha=alpha)
                    ax.plot(sv[0], (p-p0)[0], marker='>', color=color, alpha=alpha)
                    ax.plot(sv[1], (p-p0)[1], marker='x', color=color, alpha=alpha)
                    ax.plot(sv[2], (p-p0)[2], marker='o', color=color, alpha=alpha)
                i+=1

            ax.set_xlabel('$s_a, s_b$')
            ax.set_ylabel(r'$P^*-P^*_{c=0}$')
            rtypes = {'r': 'rewiring', 'g': 'growing'}
            ax.title.set_text(f'P-mat {rtypes[rtype]},\n'+
                    r'$c\in$'+f'{c_range}; '+r'$n_a=$'+f'{na}')
            #fig.colorbar(clr, ax=axs[-1], shrink=0.6)
    fig.tight_layout()
    fig.savefig(f'fig4_tests/pmat_compound_na_effect_colmin_{rtypes[rtype]}.pdf')

def plot_fig4_slines(sdata=None, spath='fig4_tests'):
    """
    Main plot for Figure 4, which combines the five datasets and divides between growing and rewiring data (left and right, respectively), and computes the theoretical effect of preferential attachment in the fixed point.
    """
    if not sdata:
        sdata = fit_all_sets()
    sdata = extra_filter(sdata)
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 8), gridspec_kw={'width_ratios': [1, 1.1]})
    rtypes = {'aps': 'g', 'airport': 'r', 'twitter': 'r', 'boards': 'r', 'isi': 'g'}
    axs[1, 0].axhline(0, color='grey', alpha=.4)
    axs[1, 1].axhline(0, color='grey', alpha=.4)
    for dset, data in sdata.items():
        rtype = rtypes[dset]
        al = True if dset == 'aps' else False
        if rtype == 'g':
            plot_slines_rho(data, axs[0, 0], add_lbl=al)
            plot_slines_ptheo(data, axs[1, 0], add_lbl=al)
        elif rtype == 'r':
            plot_slines_rho(data, axs[0, 1], add_lbl=al)
            plot_slines_ptheo(data, axs[1, 1], add_lbl=al)

    axs[0, 0].legend(title='Group ' + r'$g=$')
    cbar = fig.colorbar(cm.ScalarMappable(cmap='coolwarm'), ax=axs[:, 1], shrink=.5, label=r'$c$')
    cbar.outline.set_visible(False)
    axs[0, 0].set_title('Growing')
    axs[0, 1].set_title('Rewiring')
    for ax in axs.reshape(-1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    #fig.tight_layout()
    axs[0, 0].set_ylabel('Observed normalized group density,\n'+r'$\hat{\rho}_g$')
    axs[1, 0].set_ylabel('Effect of ' + r'$c$' + ' on fixed P matrix,\n'+r'$P^* - P^*_{c=0}$')
    fig.savefig(f'{spath}/fig4_slines_tst.pdf')

def sdata_to_table():
    sdata = fit_all_sets()

    sdata = extra_filter(sdata)
    cols = ['set', 'a', 'b', 'name', 'L', 'N', 'na', 'paa', 'pbb', 'pab', 'core', 'cp_meas', 'c', 'sa', 'sb', 'chi']
    df0 = pd.DataFrame()
    subs = {}
    for col in cols:
        subs[col] = []
        for dset, vals in sdata.items():
            for val in vals:
                if isinstance(val[col], float):
                    val[col] = np.round(val[col], 3)
                subs[col].append(val[col])
    df = pd.DataFrame(subs)
    df.to_csv('sdata.csv', index=False, sep=',')
    return df


def extra_filter(sdata):
    """
    Manage filtering data for Figure 4 //
    """
    sdata2 = {}
    for dset, data in sdata.items():
        data2 = []
        if dset in ['airport', 'twitter']:
            for dt in data:
                cp = dt['cp_meas']
                if cp > .1 and dt.get('DT', 1296001) > 1296000:
                    data2.append(dt)
        #if dset=='isi':
        #    pairs = ['{}_{}'.format(d['a'], d['b']) for d in data]
        #    for name in np.unique(pairs):
                # remove repated data
        #        vals = [(d['yrs'], min([np.abs(d['sa']-.5), np.abs(d['sb']-.5)])) for d in data if name in d['name']]
        #        vals = sorted(vals, key=lambda x: x[1])
        #        bval = vals[0][0]
        #        for d in data:
        #            if name in d['name'] and bval in d['yrs']:
        #                data2.append(d)
        elif dset in ['aps', 'boards', 'isi']:
            data2 = data
        sdata2[dset] = data2
    return sdata2


def plot_slines_rho(data, ax, color='', add_lbl=False):
    add_lbl0 = add_lbl
    for dat in data:
        na = dat['na']
        if na < .49:
            sa = dat['sa']; sb = dat['sb']
            ra = dat['rhoa']; rb = dat['rhob']; rab = dat['rhoab']
            #rt0 = dat['f_rho'][0]
            #ra0, rab0, rb0 = rt0
            r = ra + rb + rab
            c = dat['c']
            rhov = [ra/r, rab/r, rb/r]; sv = [sa, .5*(sa+sb), sb]
            color = cm.coolwarm(c)

            markers = [('>', r'$a$'), ('x', r'$ab$'), ('o', r'$b$')]
            ax.plot(sv, rhov, c=color, alpha=.2)
            for i, (marker, lab) in enumerate(markers):
                label = lab if add_lbl else ''
                ax.plot(sv[i], rhov[i], c=color, marker=marker, label=label, linestyle=None)
            add_lbl = False
    ax.set_xlabel('Homophily, '+r'$s_g$')
    #ax.set_ylabel('Observed normalized group density,\n'+r'$\hat{\rho}_g$')

def plot_slines_ptheo(data, ax, color='', add_lbl=False):
    for dat in data:
        na = dat['na']
        if na < .49:
            sa = dat['sa']; sb = dat['sb']
            rho = dat['rho']
            rt0 = dat['f_rho1'][0]
            rt = dat['f_rho'][0]
            c = dat['c']
            p_t = np.array(rho_to_p(rt, rho, na))
            p_0 = np.array(rho_to_p(rt0, rho, na))
            pv = p_t - p_0; sv = [sa, .5*(sa+sb), sb]
            color = cm.coolwarm(c)
            markers = [('>', r'$a$'), ('x', r'$ab$'), ('o', r'$b$')]

            ax.plot(sv, pv, c=color, alpha=.2)
            for i, (marker, lab) in enumerate(markers):
                label = lab if add_lbl else ''
                ax.plot(sv[i], pv[i], c=color, marker=marker, label=label)
            add_lbl = False
    ax.set_xlabel('Homophily, '+r'$s_g$')
    #ax.set_ylabel('Effect of ' + r'$c$' + ' on fixed P matrix,\n'+r'$P^* - P^*_{c=0}$')

def plot_baseline_ccol(plot_name='slines_rho', sdata=None, spath='fig4_tests'):
    """
    Baseline plot where color represents c
    """
    if not sdata:
        sdata = fit_all_sets()
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4), gridspec_kw={'width_ratios': [1, 1.1]}, sharey=True)
    rtypes = {'airport': 'r', 'twitter': 'r', 'boards': 'r', 'isi': 'g', 'aps': 'g'}
    line = np.linspace(0, 1, 100)
    axs[0].axhline(0, color='grey', alpha=.4)
    axs[1].axhline(0, color='grey', alpha=.4)
    #cmap = cm.tab20b
    #colors = {'airport': cmap(.05), 'twitter': cmap(.25), 'boards': cmap(.45), 'isi': cmap(.65), 'aps': cmap(.85)}
    al = True
    for dset, data in sdata.items():
        rtype = rtypes[dset]
        if rtype == 'g':
            eval(f'plot_{plot_name}(data, axs[0], add_lbl=al)')
        elif rtype == 'r':
            eval(f'plot_{plot_name}(data, axs[1], add_lbl=al)')
        al = False

    axs[1].legend(title='Group ' + r'$g=$')
    cbar = axs[1].figure.colorbar(cm.ScalarMappable(cmap='coolwarm'), label=r'$c$')
    cbar.outline.set_visible(False)
    axs[0].set_title('Growing')
    axs[1].set_title('Rewiring')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{spath}/{plot_name}.pdf')


def plot_baseline_params(plot_name='c_amp_theo', sdata=None, spath='fig4_tests'):
    """
    Baseline plot where x is some param and y some observed/theo metric
    """
    if not sdata:
        sdata = fit_all_sets()
    fig, ax = plt.subplots(figsize=(4.4, 4))
    rtypes = {'airport': 'r', 'twitter': 'r', 'boards': 'r', 'isi': 'g', 'aps': 'g'}
    line = np.linspace(0, 1, 100)
    ax.axhline(0, color='grey', alpha=.4)
    cmap = cm.tab20b; al=True
    colors = {'airport': cmap(.05), 'twitter': cmap(.25), 'boards': cmap(.45), 'isi': cmap(.65), 'aps': cmap(.85)}
    for dset, data in sdata.items():
        marker = 'x' if rtypes[dset]=='r' else 'o'
        color = colors[dset]
        eval(f'plot_{plot_name}(data, ax, add_lbl=al, marker=marker, color=color)')
        al = False

    ax.legend(title='Dataset')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{spath}/{plot_name}.pdf')

def plot_c_amp_theo(data, ax, color='', marker='x', add_lbl=False):
    frst = True
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            cp, cp0 = dat['f_cp_meas'], dat['f_cp_meas1']
            amp = (cp-cp0) / cp0
            c = dat['c'] + np.random.uniform(-.02, .02)
            label = dat['set'] if frst else ''
            ax.plot(c, amp, marker=marker, color=color, label=label)
            frst = False
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$\frac{CP-CP_{c=0}}{CP_{c=0}}$')
    ax.set_yscale('log')
    ax.set_title('CP amplification with c')


def plot_c_cpobs(data, ax, color='', marker='x', add_lbl=False):
    frst = True
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            cp_obs, cp_theo = dat['cp_meas'], dat['f_cp_meas']
            amp = (cp_obs - cp_theo) #/ cp_theo
            c = dat['c'] + np.random.uniform(-.02, .02)
            label = dat['set'] if frst else ''
            ax.plot(c, cp_obs, marker=marker, color=color, label=label)
            frst = False
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$CP_{obs}$')
    #ax.set_yscale('log')


def plot_c_cprho(data, ax, color='', marker='x', add_lbl=False):
    frst = True
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            cp_obs, cp_theo = dat['cp_meas'], dat['f_cp_meas']
            core = dat['core']
            if core:
                prif = 'b' if core == 'a' else 'a'
                rhoc = dat['rho'+core]
                rhop = dat['rho'+prif]
                rhocp = dat['rhoab']

                rhot = rhoc+rhop+rhocp; dat['rho']
                ftr = na if core == 'a' else 1-na
                cp_f = (rhoc/rhot) * cp_obs
                c = dat['c']+np.random.uniform(-.02, .02)
                label = dat['set'] if frst else ''
                ax.plot(c, cp_f, marker=marker, color=color, label=label)
                frst = False
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$CP^{\rho}_{obs}=\hat{\rho}_c*\frac{\rho_p-\rho_{cp}}{\rho_p+\rho_{cp}}$')
    ax.set_title('Observed ' + r'$CP^{\rho}$')


def plot_c_cprhodiff(data, ax, color='', marker='x', add_lbl=False):
    frst = True
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            cp_obs, cp_theo = dat['cp_meas'], dat['f_cp_meas']
            core = dat['core']
            if core:
                prif = 'b' if core == 'a' else 'a'
                rhoc = dat['rho'+core]
                rhop = dat['rho'+prif]
                rhocp = dat['rhoab']

                rhot = rhoc+rhop+rhocp; dat['rho']
                ftr = na if core == 'a' else 1-na
                cp_f = (rhoc/rhot) * cp_obs

                rhoc_t = dat['f_rho'][0][0] if core=='a' else dat['f_rho'][0][2]
                cp_t = (rhoc_t/sum(dat['f_rho'][0]))*cp_theo
                c = dat['c']+np.random.uniform(-.02, .02)
                label = dat['set'] if frst else ''
                ax.plot(c, cp_t-cp_f, marker=marker, color=color, label=label)
                frst = False
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$CP^{\rho}_{*}-CP^{\rho}_{obs}$')


def plot_c_cptheo(data, ax, color='', marker='x', add_lbl=False):
    frst = True
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            cp_meas, cp_meas0 = dat['f_cp_meas'], dat['f_cp_meas1']
            core = dat['core']
            if core:
                rhoc_t = dat['f_rho'][0][0] if core=='a' else dat['f_rho'][0][2]
                rhoc_t0 = dat['f_rho1'][0][0] if core=='a' else dat['f_rho1'][0][2]

                cp_t = (rhoc_t/sum(dat['f_rho'][0])) * cp_meas
                cp_t0 = (rhoc_t0/sum(dat['f_rho1'][0])) * cp_meas0
                c = dat['c']+np.random.uniform(-.02, .02)
                label = dat['set'] if frst else ''
                ax.plot(c, cp_t-cp_t0, marker=marker, color=color, label=label)
                frst = False
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$CP^{\rho} - CP^{\rho}_{c=0}$')
    ax.set_title('Effect of c on '+ r'CP^{\rho}')

def plot_comp_homc(data, ax, color='', marker='x', add_lbl=False):
    frst = True
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .5:
            sa = dat['sa'] + np.random.uniform(-.02, .02)
            sb = dat['sb'] + np.random.uniform(-.02, .02)
            label = dat['set'] if frst else ''
            ax.plot(sa, sb, marker=marker, color=color, label=label)
            frst = False
    ax.set_xlabel(r'$s_a$')
    ax.set_ylabel(r'$s_b$')
    ax.set_title('Estimated homophilies')



def plot_slines_rhotheo(data, ax, color='', add_lbl=False):
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            sa = dat['sa']; sb = dat['sb']
            rt = dat['f_rho'][0]
            ra, rab, rb = rt
            r = ra + rb + rab
            c = dat['c']
            rhov = [ra/r, rab/r, rb/r]; sv = [sa, .5*(sa+sb), sb]
            color = cm.coolwarm(c)

            ax.plot(sv, rhov, c=color)
            markers = [('>', r'$a$'), ('x', r'$ab$'), ('o', r'$b$')]
            for i, (marker, lab) in enumerate(markers):
                label = lab if add_lbl else ''
                ax.plot(sv[i], rhov[i], c=color, marker=marker, label=label)
            add_lbl = False
    ax.set_xlabel('Homophily, '+r'$s_g$')
    ax.set_ylabel('Theoretical group density, '+r'$\rho_g$')



def plot_slines_pobs(data, ax, color='', add_lbl=False):
    for dat in data:
        na = dat['na']
        if np.abs(na-.5) < .49:
            sa = dat['sa']; sb = dat['sb']
            rho = dat['rho']
            rt0 = dat['f_rho1'][0]
            c = dat['c']
            p_obs = np.array((dat['paa'], 1-dat['paa']-dat['pbb'], dat['pbb']))
            p_0 = np.array(rho_to_p(rt0, rho, na))
            pv = p_obs - p_0; sv = [sa, .5*(sa+sb), sb]
            color = cm.coolwarm(c)
            markers = [('>', r'$a$'), ('x', r'$ab$'), ('o', r'$b$')]

            ax.plot(sv, pv, c=color, alpha=.2)
            for i, (marker, lab) in enumerate(markers):
                label = lab if add_lbl else ''
                ax.plot(sv[i], pv[i], c=color, marker=marker, label=label)
            add_lbl = False
    ax.set_xlabel('Homophily, '+r'$s_g$')
    ax.set_ylabel('Effect of ' + r'$c$' + ' on observed P matrix,\n'+r'$P^n - P^*_{c=0}$')

def rho_to_p(rhot, rho, na):
    paa = rhot[0]*na**2/rho
    pbb = rhot[2]*(1-na)**2/rho
    return paa, 1-paa-pbb, pbb


def dat_to_fixp(dat, rtype):
    rho = dat.get('rho', 2*dat['L']/(dat['N']**2))
    dat['rho'] = rho
    c, sa, sb, na = dat['c'], dat['sa'], dat['sb'], dat['na']
    sa = min([1, sa]); c = min([1, c]); sb = min([1, sb])
    sa0, sb0 = dat['sa0'], dat['sb0']
    sa0 = min([1, sa0]); sb0 = min([1, sb0])

    frhoc = verify_fixp(c=c, sa=sa, sb=sb, na=na, rho=rho, rtype=rtype)
    frho0 = verify_fixp(c=.001, sa=sa0, sb=sb0, na=na, rho=rho, rtype=rtype)
    frho1 = verify_fixp(c=.001, sa=sa, sb=sb, na=na, rho=rho, rtype=rtype)

    return frhoc, frho0, frho1

def verify_fixp(c, sa, sb, na, rho, rtype='r'):
    """
    Some fixed points might not be valid for the choice of rho (local rho > 1), in these cases, we evolve models while vality conditions are met
    """
    if rtype == 'r':
        fps, _, _ = gfp.rewire_fixed_points_density(c=c, sa=sa, sb=sb, na=na, rho=rho)
    elif rtype == 'g':
        fps, _, _ = gfp.growth_fixed_points_density(c=c, sa=sa, sb=sb, na=na, rho=rho)
    fps_n = []
    for rho_i in fps:
        val = np.all([True if r <= 1 else False for r in rho_i])
        rho_n = rho_i
        if not val:
            rho_n = ()
            paa_up =  na**2/rho
            pbb_up =  (1-na)**2/rho
            p_opt = np.array([rho_i[0]*na**2/rho, rho_i[2]*(1-na)**2/rho])
            pas = np.linspace(0, min([.95*paa_up, 1]), 7)
            pbs = np.linspace(0, min([.95*pbb_up, 1]), 7)
            Ps = [(pa, pb) for pa, pb in product(pas, pbs) if pa+pb<1]
            dist=np.inf
            for p0 in Ps:
                if rtype == 'r':
                    ysol = gfp.rewire_path(c, na, sa, sb, p0, np.linspace(0, 1000, 10000))
                else:
                    ysol = gfp.growth_path(c, na, sa, sb, p0, 100, np.linspace(0, 50000, 10000))
                idx_a = ysol[ysol[:, 0] < paa_up].shape[0] - 1
                idx_b = ysol[ysol[:, 1] < pbb_up].shape[0] - 1
                idx = min([idx_a, idx_b])
                p_val = ysol[idx]; p_fix = ysol[-1, :]
                dist_p = np.sum((p_fix - p_opt)**2)**(1/2)
                if (dist_p < dist) and idx > 0:
                    rho_n = gfp.p_to_rho([p_val], rho, na)[0]
                    dist = dist_p
        fps_n.append(rho_n)
    return fps_n


def dens_to_cpmeas(dens):
    """
    dens is (rhoa, rhoab, rhob)
    """
    dens = [min(r, 1) for r in dens]
    rhoa, rhoab, rhob = dens
    if rhoa > max([rhoab, rhob]): # and (rhoab - rhob)/(rhoab+rhob) > lwr:
        cp_meas =  (rhoab - rhob)/(rhoab+rhob)
        core =  'a'
    elif rhob > max([rhoab, rhoa]): # and (rhoab-rhoa)/(rhoab+rhoa) > lwr:
        cp_meas =  (rhoab - rhoa)/(rhoab+rhoa)
        core =  'b'
    else:
        cp_meas = np.abs(rhoa-rhob) / (rhoa+rhob)
        core = 'n'
    return cp_meas, core



def plot_diffr_x(sets_data=[], xval='diff_s', a=.01, cmap='Dark2', yval='r', figax=()):
    """
    Function for plotting three different measures of CP (yvals: r, rho, rho_bin) as a function of estimated parameters
    """
    if not sets_data:
        sets_data = []
        for dset in ['airport', 'twitter', 'isi', 'aps', 'boards']:
            fdata = eval(f'read_{dset}()')
            sets_data.append(fdata)
    if not figax:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig, ax = figax
    cmap = plt.get_cmap(cmap)
    sign = ['p', 'o', 'x', 's', '^']

    for i, fdata in enumerate(sets_data):
        color = cmap(i/5)
        sig = sign[i]
        print(color)
        if yval == 'r':
            ylabel = _plot_diffr_x_set(fdata, xval, a, color, ax, sig)
        elif yval == 'rho':
            ylabel = _plot_diffrho_x_set(fdata, xval, color, ax, sig)
        elif yval == 'rho_bin':
            ylabel = _plot_diffrhobin_x_set(fdata, xval, color, ax, sig)
    labels = {'diff_s': r'$s_a-s_b$', 'c': r'$c$', 'na': r'$n_a$'}
    ax.set_xlabel(labels[xval])
    ax.set_ylabel(ylabel)
    if yval == 'r':
       ax.set_ylim(top=2)
    if xval == 'diff_s' and yval != 'rho_bin':
        ax.set_xlim(-1, 1)
        ax.vlines(0, -1, 2, ls='--', color='darkgrey', alpha=.5)
    #ax.legend()
    #fig.tight_layout()
    return fig, ax

def plot_csna(yval='r'):
    fig, ax = plt.subplots(1, 3, figsize=(3*3, 3))
    for i, xval in enumerate(['c', 'diff_s', 'na']):
        plot_diffr_x(xval=xval, yval=yval, figax=(fig, ax[i]))
    ax[i].legend()
    fig.tight_layout()
    fig.savefig(f'diffr_{yval}.pdf')


def _plot_diffr_x_set(fdata, xval, a, color, ax, sig):
    x, y = [], []
    for d in fdata:
        label = d['set']
        ra = cpm.r_cont_norm(d['paa'], d['pab'], d['rho'], d['na'], a)
        rb = cpm.r_cont_norm(d['pbb'], d['pab'], d['rho'], 1-d['na'], a)
        if xval == 'diff_s':
            y.append(ra - rb)
            x.append(d['sa'] - d['sb'])
            ylabel = r'$\hat{r}_a - \hat{r}_b$'
        elif xval == 'c':
            y.append(np.abs(ra-rb))
            x.append(d['c'])
            ylabel = r'$|\hat{r}_a - \hat{r}_b|$'
        elif xval == 'na':
            y.append(ra - rb)
            x.append(d['na'])
            ylabel = r'$\hat{r}_a - \hat{r}_b$'
        else:
            raise('invalid x')
    ax.plot(x, y, sig, label=label, color=color) #, color)
    #ax.text(x, y, txt) #, color)
    return ylabel


def _plot_diffrho_x_set(fdata, xval, color, ax, sig):
    x, y = [], []
    for d in fdata:
        label = d['set']
        na = d['na']; rho = d['rho']
        ra = d['paa'] * rho / (na**2)
        rb = d['pbb'] * rho / (1-na)**2
        rab = ra + rb
        if xval == 'diff_s':
            y.append((ra - rb)/rab)
            x.append(d['sa'] - d['sb'])
            ylabel = r'$\frac{\rho_a - \rho_b}{\rho_a+\rho_b}$'
        elif xval == 'c':
            y.append(np.abs(ra-rb)/rab)
            x.append(d['c'])
            ylabel = r'$\frac{|\rho_a - \rho_b|}{\rho_a+\rho_b}$'
        elif xval == 'na':
            y.append((ra - rb)/rab)
            x.append(d['na'])
            ylabel = r'$\frac{\rho_a - \rho_b}{\rho_a+\rho_b}$'
        else:
            raise('invalid x')
    ax.plot(x, y, sig, label=label, color=color) #, color)
    return ylabel

def _plot_diffrhobin_x_set(fdata, xval, color, ax, sig):
    x, y = [], []
    for d in fdata:
        label = d['set']
        na = d['na']; rho = d['rho']
        ra = d['paa'] * rho / (na**2)
        rb = d['pbb'] * rho / (1-na)**2
        rab = (1-d['paa']-d['pbb'])*rho / (na*(1-na))

        rt = ra + rb + rab
        if ((ra > rab) and (rab) > (rb)) or ((rb > rab) and (rab > ra)):
            y.append(1)
        else:
            y.append(0)
        if xval == 'diff_s':
            x.append(d['sa'] - d['sb'])
        elif xval == 'c':
            x.append(d['c'])
        elif xval == 'na':
            x.append(d['na'])
        else:
            raise('invalid x')
    ylabel = 'Binary CP'
    ax.plot(x, y, sig, label=label, color=color) #, color)
    return ylabel


def plot_diffr_c_sample(sets_data=[], xval='diff_s', a=.01, cmap='Dark2', yval='r', figax=()):
    if not sets_data:
        sets_data = []
        for dset in ['airport', 'twitter', 'isi', 'aps', 'boards']:
            fdata = eval(f'read_{dset}()')
            sets_data.append(fdata)
    if not figax:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig, ax = figax
    cmap = plt.get_cmap(cmap)
    sign = ['p', 'o', 'x', 's', '^']

    for i, fdata in enumerate(sets_data):
        import pdb; pdb.set_trace()
        color = cmap(i/5)
        sig = sign[i]
        print(color)
        ylabel = _plot_diffr_x_set(fdata, xval, a, color, ax, sig)


    labels = {'diff_s': r'$s_a-s_b$', 'c': r'$c$', 'na': r'$n_a$'}
    ax.set_xlabel(labels[xval])
    ax.set_ylabel(ylabel)
    if yval == 'r':
       ax.set_ylim(top=2)
    if xval == 'diff_s' and yval != 'rho_bin':
        ax.set_xlim(-1, 1)
        ax.vlines(0, -1, 2, ls='--', color='darkgrey', alpha=.5)

    return fig, ax


def plot_miniplots_diffr_c():
    left, bottom, width, height = [0.32, 0.62, 0.175, 0.175]
    axm1 = fig.add_axes([left, bottom, width, height])
    s1 = .625
    sol1 = gfp.rewire_fixed_points(na=.5, c=.9, sa=s1, sb=s1)
    r1a = gfp.cp_correlation_cont(*sol1[2], na=.5, rho=.1, a=.01)
    r1b = gfp.cp_correlation_cont(sol1[2][1], sol1[2][0] , na=.5, rho=.1, a=.01)
    #ax.plot(s1, r1a-r1b, 's', color='red')
    plot_mini_net(*sol1[2], ax=axm1)


def get_miniplot_sample_data():
    sets_data = []
    for dset in ['airport', 'twitter', 'isi', 'aps', 'boards']:
        fdata = eval(f'read_{dset}()')
        sets_data.append(fdata)
    net_dat = []
    a = .01
    for fdata in sets_data:
        for d in fdata:
            label = d['set']
            na = d['na']; rho = d['rho']
            rhoa = d['paa'] * rho / (na**2)
            rhob = d['pbb'] * rho / (1-na)**2

            ra = cpm.r_cont_norm(d['paa'], d['pab'], d['rho'], d['na'], a)
            rb = cpm.r_cont_norm(d['pbb'], d['pab'], d['rho'], 1-d['na'], a)

            rval = np.abs(ra - rb)
            rhoval = np.abs(rhoa-rhob) / (rhoa + rhob)
            N = d.get('N', 1000)
            net_dat.append([label, d['name'], rval, rhoval, d['paa'], d['pbb'], N, rho, na])
    net_df = pd.DataFrame(net_dat, columns=['dset', 'name', 'rval', 'rhov', 'paa', 'pbb', 'N', 'rho', 'na'])
    return net_df

def plot_mininet_idx(idx, ax=None):
    net_df = get_miniplot_sample_data()
    dat = net_df.iloc[idx]
    N = int(dat.N/50) #
    N = min([dat.N, 10000])
    plot_mini_net(paa=dat.paa, pbb=dat.pbb, rho=dat.rho, na=dat.na, N=N, ax=ax)


def plot_boards_net(ax=None, node_size=10, start=2007, end=2009, dt=12):
    from fit_boards import get_data_in_range, parse_groups
    _, _, _, net_og, obs = get_data_in_range(start=start, end=end, dt=dt, return_net=True)
    groups = parse_groups()
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2))
    options = {'node_size': node_size, 'ax':ax, 'alpha': .4}
    ccnodes = max(nx.connected_components(net_og), key=len)
    net = net_og.subgraph(ccnodes)
    pos = nx.spring_layout(net) #, seed=1235)

    nx.draw_networkx_edges(net, pos, edge_color='grey', alpha=.3, ax=ax)
    anodes = [i for i in net.nodes() if groups[i] == '2']
    bnodes = [i for i in net.nodes() if groups[i] == '1']
    nx.draw_networkx_nodes(net, pos, nodelist=bnodes, node_shape='<', node_color='royalblue', **options)
    nx.draw_networkx_nodes(net, pos, nodelist=anodes, node_shape='s', node_color='orangered', **options)
    for lab in ['top', 'bottom', 'left', 'right']:
        ax.spines[lab].set_visible(False)


# Subplot fig4
def plot_airport(ax=None, node_size=10, a=[5], b=[9], years=(1999, 2000), dt=4):
    from fit_airport_db1b import RouteEvol
    RE = RouteEvol(a, b, years, dt)
    net_og = RE.net
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2))
    options = {'node_size': node_size, 'ax':ax, 'alpha': .4}
    ccnodes = max(nx.connected_components(net_og), key=len)
    net = net_og #.subgraph(ccnodes)

    anodes = [i for i in net.nodes() if RE.get_group(i) == 'a']
    print(len(anodes))
    subnet_a = net.subgraph(anodes)
    pos_a = nx.spring_layout(subnet_a)
    pos_a = {k: v / 4 for k, v in pos_a.items()}
    bnodes = [i for i in net.nodes() if RE.get_group(i) == 'b']
    print(len(bnodes))

    pos = nx.spring_layout(net, pos=pos_a, fixed=anodes) #, seed=1235)
    nx.draw_networkx_edges(net, pos, edge_color='grey', alpha=.3, ax=ax)
    nx.draw_networkx_nodes(net, pos, nodelist=bnodes, node_shape='<', node_color='royalblue', **options)
    nx.draw_networkx_nodes(net, pos, nodelist=anodes, node_shape='s', node_color='orangered', **options)
    for lab in ['top', 'bottom', 'left', 'right']:
        ax.spines[lab].set_visible(False)

def plot_twitter_net(ax=None, node_size=10, a=[5], b=[9], years=(1999, 2000), dt=4):
    from plot_twitter import parse_groups, get_net0, read_logs, logpath
    df = read_logs(logpath, date0='2020-10-12', daten='2020-11-12')
    net, _, _ = get_net0(df)

    groups = parse_groups()
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2))
    options = {'node_size': node_size, 'ax':ax, 'alpha': .5}
    degs = sorted([net.degree(i) for i in net.nodes])
    max_deg = degs[int(len(degs)*.50)]
    r_nodes = []
    for node in net.nodes():
        if (net.degree(node) > max_deg) and (node not in groups):
            r_nodes.append(node)
    net.remove_nodes_from(r_nodes)

    pos = nx.spring_layout(net) #, seed=1235)

    nx.draw_networkx_edges(net, pos, edge_color='grey', alpha=.3, ax=ax)
    anodes = [i for i in net.nodes() if i in groups]
    bnodes = [i for i in net.nodes() if i not in groups]
    nx.draw_networkx_nodes(net, pos, nodelist=bnodes, node_shape='<', node_color='royalblue', **options)
    nx.draw_networkx_nodes(net, pos, nodelist=anodes, node_shape='s', node_color='orangered', **options)
    for lab in ['top', 'bottom', 'left', 'right']:
        ax.spines[lab].set_visible(False)

def plot_c_snapshots(sets_data=[], cmap='Dark2', figax=()):
    """
    Function for plotting three snapshots of c (low, med, high), and at each snapshot plotting core homophily, measure of CP and other homophily )
    """
    if not sets_data:
        sets_data = []
        for dset in ['airport', 'twitter', 'isi', 'aps', 'boards']:
            fdata = eval(f'read_{dset}()')
            sets_data.append(fdata)
    if not figax:
        fig, axs = plt.subplots(1, 3, figsize=(3*4, 4))
    else:
        fig, axs = figax
    cmap = plt.get_cmap(cmap)
    sign = ['p', 'o', 'x', 's', '^']

    for i, fdata in enumerate(sets_data):
        color = cmap(i/5)
        sig = sign[i]
        ylabel = _plot_c_snapshots(fdata, color, axs, sig)

    labels = {'diff_s': r'$s_a-s_b$', 'c': r'$c$', 'na': r'$n_a$'}
    axs[0].set_ylabel(r'$\frac{\rho_c - \rho_p}{\rho_c+\rho_p}$')
    axs[0].set_xlabel(r'$c$')
    #ylab = r'$\frac{s_c-s_p}{s_c}$'
    ylab = r'$s_c$'
    axs[1].set_xlabel(ylab)
    axs[2].set_xlabel(ylab)
    axs[0].title.set_text('PA and CP')
    axs[1].title.set_text('Homophily and CP: Low PA')
    axs[2].title.set_text('Homophily and CP: High PA')
    #ax.set_ylabel(ylabel)
    #if yval == 'r':
    #   ax.set_ylim(top=2)
    #if xval == 'diff_s' and yval != 'rho_bin':
    #    ax.set_xlim(-1, 1)
    #    ax.vlines(0, -1, 2, ls='--', color='darkgrey', alpha=.5)
    #ax.legend()
    fig.tight_layout()
    #fig.savefig('fig4_tests/onion_relative-homophily.pdf')
    fig.savefig('fig4_tests/onion_homophily.pdf')
    return fig, axs


def _plot_c_snapshots(fdata, color, axs, sig):
    for d in fdata:
        label = d['set']
        na = d['na']; rho = d['rho']
        cm = d.get('cp_meas', None)
        if label in ['airport', 'twitter', 'board']:
            shape = 'o'
        elif label in ['aps', 'isi']:
            shape = 'X'
        x0 = d['c']
        y = np.nan #(rhoa-rhob)/(rhoa+rhob)
        if not cm:
            rhoa = d['paa'] * rho / (na**2)
            rhob = d['pbb'] * rho / ((1-na)**2)
            if label == 'twitter':
                rhoab = (d['pba']+d['pab'])* rho / (2*na*(1-na))
            else:
                rhoab = d['pab'] * rho / (na*(1-na))
            if rhoa > max([rhob, rhoab]):
                y = (rhoab-rhob) / (rhob+rhoab)
                x = d['sa'] #(d['sa'] - d['sb'] ) / d['sa']
                #shape = 'o'
            elif rhob > max([rhoa, rhoab]):
                y = (rhoab-rhoa) / (rhoab+rhoa)
                x = d['sb'] #(d['sb'] - d['sa'] ) / d['sb']
                #shape = 'o'
            if x > 1:
                x = np.nan
        else:
            y = cm
            x0 = d['c']
            sc, sp = (d['sa'], d['sb']) if d['core'] == 'a' else (d['sb'], d['sa'])
            x = (sc - sp) / sc
            x = sc

        axs[0].plot(x0, y, shape, color=color)
        if (x0 >=  2/3):
            c_cat = 2
        elif x0 <= 1/3:
            c_cat = 1
        else:
            c_cat = None
        if c_cat:
            axs[c_cat].plot(x, y, shape, color=color)



if __name__ == '__main__':
    #import argparse as args
    plot_c_snapshots()
    #parser = args.ArgumentParser()
    #parser.add_argument('--plot')

