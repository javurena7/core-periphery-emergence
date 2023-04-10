import get_fixed_points as gfp
import matplotlib.pyplot as plt; plt.ion()
plt.rcParams["axes.grid"] = False
import seaborn as sns
import numpy as np
from itertools import product
import pickle as p
from scipy import interpolate
import os
from copy import deepcopy


lab_dict = {'sa': r'$s_a$', 'sb': r'$s_b$', 'c': r'$c$', 'na': r'$n_a$', 'rho':r'$\rho$'}

def plot_amplification_acore_rewire(n_c=6, n_size=100, xlims=(0, 1), data=None, sim_data=None, Nnodes=1500, n_sim=5, figax=(), miniplots=False):
    """
    Main plot for figure 2, where sa=sb, na=.5 and we test differet c values
    IF sim_data is True, we get sim_data // otherweise we can pass the sim_data dict
    n_c: number of c values (from .5 to 1)
    n_size: number of observation points for homophily

    """
    data = get_amplification_acore_rewire(n_c=n_c, n_size=n_size, xlims=xlims) #(.49, 1))
    path = f'amp/acore_c{n_c}_rewire_xlims{xlims[0]}-{xlims[1]}.pdf'
    if not figax:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = figax
    ax.set_xlim(0, 1)
    xvals = data['xvals']
    cvals = data['cvals']
    if sim_data is True:
        sim_path = 'sim/equal_homophily_N{Nnodes}_t2000.p'
        params = {'na': .5, 'rho': .1}
        sim_data = get_sim_data_rewire(x='s', params=params, xlims=xlims, cvals=cvals, n_sim=n_sim, Nnodes=Nnodes, path=sim_path)
    colors = plt.cm.BuGn(cvals)
    ax.axhline(xmin=0, xmax=.5, color='k')
    ax.axhline(xmin=0.5, xmax=1, color='k', ls=':')
    for i, cv in enumerate(data['cvals']):
        ax.plot(xvals, data['stable1'][i, :], color=colors[i], label='c={}'.format(np.round(cv, 2)))
        ax.plot(xvals, data['stable2'][i, :], color=colors[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'${s=s_a=s_b}$')

    if sim_data:
        sim_cvals = sorted(np.unique([v['c'] for v in sim_data.values()]))
        cols = {np.round(cv, 2): colors[i] for i, cv in enumerate(cvals)}
        for pt in sim_data.values():
            xt = 'sa' #x if x in ['sa', 'sb', 'na', 'c'] else 'sa'
            xv = pt[xt]; Pv = pt['P']; cv = pt['c']; nav = pt['na']; rhov = pt['rho']
            ra = gfp.cp_correlation_cont(*Pv, nav, rhov, 0.01)
            rb = gfp.cp_correlation_cont(Pv[1], Pv[0], nav, rhov, 0.01)
            yv = ra - rb
            color = cols[np.round(cv, 2)]
            ax.plot(xv, yv, 'x', color=color)
    if miniplots:
        plot_miniplots_acore_rewire(fig, ax)
    ax.set_ylabel(r'$r_a-r_b$')
    #ax.set_title('Core-periphery amplification')
    ax.legend()
    fig.savefig(path)

def plot_phase_space_rewire_combined(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(0, 1), ylims=(0, 1), n_size=150, fig=None, ax=None, n_ticks=6, data=(), border_width=9, title='', cbar_ax=None, cmap='mako', segment=True, extraname='', lev=1):
    """
    Phase Space heatmaps for rewiring model (Figure 3) combining color BE'r and contour omega_cp.
    Plots two areas: (red) number of solutions (either one or three),
        (grey): density-dominant areas (either a or b)
    Combines two heatmaps within the density-dominant area: (countour) omega
        (color) Borgatti's r

    Variable x (y) iterates values through xlims (ylims), getting n_size evaluation points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    if not extraname:
        extraname = '_'.join([f'{k}{v}' for k, v in params.items()]) + f'_n{n_size}'
    if not data:
        datar, datao = get_phase_space_rewire_combined(x, y, params, xlims, ylims, n_size, extraname=extraname)

    # PLOT R HEATMAPS
    acore = datar['acore']; bcore = datar['bcore']
    oacore = datao['acore']; obcore = datao['bcore']
    center_mask = ~np.isnan(acore + bcore)
    center = np.maximum(acore, bcore) #.5*(acore + bcore)
    nocore = np.nan_to_num(acore) + np.nan_to_num(bcore)
    acore_mask = np.ma.masked_where(center_mask, acore)
    bcore_mask = np.ma.masked_where(center_mask, bcore)

    ax.imshow(acore_mask, vmin=-1, vmax=1, cmap=cmap)
    ax.imshow(bcore_mask, vmin=-1, vmax=1, cmap=cmap)
    pos = ax.imshow(center, vmin=-1, vmax=1, cmap=cmap)

    # PLOT OMEGA CONTOUR
    #levels = np.round(np.linspace(-1, 1, 11), 2)
    n_omega = 11
    levels = np.round(np.linspace(-.75, .75, n_omega), 2)
    ctr = int((n_omega -1) / 2)
    if not np.isnan(oacore).all():
        CA = ax.contour(oacore, levels, linestyles=':', cmap='coolwarm')
        fmta = {l: str(l) if l != 0. else r'$\Omega_{a}=0$'for l in levels}
        ax.clabel(CA, levels[::lev], fontsize=13, fmt=fmta, inline=True)
        CA.collections[ctr].set_linewidth(border_width*.8)
        CA.collections[ctr].set_linestyle('-')
    if not np.isnan(obcore).all():
        CB = ax.contour(obcore, levels, linestyles='--', cmap='coolwarm')
        fmtb = {l: str(l) if l != 0. else r'$\Omega_{b}=0$'for l in levels}
        ax.clabel(CB, levels[::lev], fontsize=13, fmt=fmtb, inline=True)
        CB.collections[ctr].set_linewidth(border_width*.8)
        CB.collections[ctr].set_linestyle('-')

    # Plot CP border
    st_border = datar['st_border']
    st_border[st_border == 1] = np.nan
    cborder = get_border(st_border)

    aborder = get_border(acore)
    bborder = get_border(bcore)

    invert = False
    if x=='na' and y=='c' and params['sa']==params['sb']:
        invert = 'y'
    if not np.isnan(aborder).all():
        aint = interpolate_border(aborder, n_size/6, invert=invert)
        ax.plot(aint[0], aint[1], linewidth=border_width, color='grey')

    if not np.isnan(bborder).all():
        bint = interpolate_border(bborder, n_size/6, invert=invert, segment=False)
        ax.plot(bint[0], bint[1], linewidth=border_width, color='grey')

    if ~np.isnan(cborder).all():
        segment = True if x=='sa' and y=='sb' else False
        cint = interpolate_border(cborder, n_size/4, segment=segment)
        ax.plot(cint[0], cint[1], linewidth=border_width, color='red', linestyle=':')

    a_loc = np.mean(np.where(acore_mask > 0), 1)
    b_loc = np.mean(np.where(bcore_mask > 0), 1)

    u_loc = np.mean(np.where(nocore == 0), 1)
    props = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    ax.text(a_loc[1], a_loc[0], 'A+', color='darkslategrey', size=12, bbox=props)
    ax.text(b_loc[1], b_loc[0], 'B+', color='darkslategrey', size=12, bbox=props)
    if ~np.isnan(cborder).all():
        c_loc = np.mean(np.where(center > 0), 1)
        ax.text(c_loc[1]-.08*n_size, c_loc[0]-.05*n_size, 'A+ B+\n  U', color='darkslategrey', size=12, bbox=props)
    ax.text(u_loc[1], u_loc[0], '0', size=12, color='darkslategrey', bbox=props)

    ax.axvline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.axhline(n_size/2, linestyle='--', alpha=.7, color='grey')

    xticks = np.linspace(0, n_size, n_ticks)
    xticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*xlims, n_ticks)]
    yticks = np.linspace(0, n_size, n_ticks)
    yticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*ylims, n_ticks)]
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels) #[(xt, xl) for xt, xl in zip(xticks, xticklabels)])
    ax.set_yticklabels(yticklabels) #[(xt, xl) for xt, xl in zip(yticks, xticklabels)])
    ax.set_xlabel(lab_dict[x])
    ax.set_ylabel(lab_dict[y])
    for loc in ['top', 'right', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)

    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    name = 'ps_plots_comb/phsp_rcomb_ppr_{}_{}_{}.pdf'.format(x, y, extraname)

    if fig:
        cbar = fig.colorbar(pos, ax=ax)
        label = r'$r_g$'
        cbar.set_label(label, size=10)
        cbar.outline.set_visible(False)
        #for loc in ['top', 'right', 'left', 'bottom']:
        #    cbar.spines[loc].set_visible(False)
        fig.tight_layout()
        fig.savefig(name)


def plot_amplification_acore_growth(n_c = 6, n_size=50, xlims=(0, 1), data=None, sim_data=None, Nnodes=1500, n_sim=5, figax=(), miniplots=False):
    """
    IF sim_data is True, we get sim_data // otherweise we can pass the sim_data dict
    """
    data = get_amplification_acore_growth(n_c=n_c, n_size=n_size, xlims=xlims)
    path = f'amp/acore_c{n_c}_growth_xlims{xlims[0]}-{xlims[1]}.pdf'
    if not figax:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig, ax = figax

    xvals = data['xvals']
    cvals = data['cvals']
    if sim_data is True:
        sim_path = f'amp/sim_acore_growth_N{Nnodes}_nc{n_c}_nsim{n_sim}_xlim{xlims[0]}-{xlims[1]}.p'
        sim_data = get_sim_data_growth(x=x, params=params, xlims=xlims, cvals=cvals, n_sim=n_sim, Nnodes=Nnodes, path=sim_path)
    colors = plt.cm.autumn(cvals)
    for i, cv in enumerate(data['cvals']):
        ax.plot(xvals, data['single'][i, :], color=colors[i], label='c={}'.format(np.round(cv, 1)), alpha=.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$s_a$')

    if sim_data:
        xvals = sim_data['xvals']
        for i, cv in enumerate(data['cvals']):
            ax.plot(xvals, sim_data['stable1'][i, :], 'x', color=colors[i])
            ax.plot(xvals, sim_data['stable2'][i, :], 'x', color=colors[i])
    if miniplots:
        plot_miniplots_growth(fig, ax)
    ax.set_ylabel(r'$r_a$')
    ax.set_title('Growth: ' + r'$s_b=0.5$')
    ax.legend()
    #fig.tight_layout()
    fig.savefig(path)


def plot_phase_space_grow_combined(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.15}, xlims=(0, 1), ylims=(0, 1), n_size=150, fig=None, ax=None, data=(), title='', n_ticks=6, border_width=8, cmap='mako', extraname='', lev=1):
    """
    Figure 3 subplots for growing model -- combining BE's r measure and densitity contour
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    if not extraname:
        extraname = '_'.join([f'{k}{v}' for k, v in params.items()]) + f'_n{n_size}'
    datar, datao = get_phase_space_grow_combined(x, y, params, xlims, ylims, n_size, extraname=extraname)
    #datao = get_phase_space_grow_combined(x, y, params, xlims, ylims, n_size, extraname=extraname, onion=True)
    acore, bcore = datar['acore'], datar['bcore']
    oacore, obcore = datao['acore'], datao['bcore']

    # PLOT R
    ax.imshow(acore, vmin=0, vmax=1, cmap=cmap)
    pos = ax.imshow(bcore, vmin=0, vmax=1, cmap=cmap)

    #ax.imshow(acore, vmin=-1, vmax=1, cmap='icefire')
    #pos = ax.imshow(bcore, vmin=-1, vmax=1, cmap='icefire')

    # PLOT OMEGA
    n_omega = 11
    levels = np.round(np.linspace(-.75, .75, n_omega), 2)
    ctr = int((n_omega-1)/2)
    if not np.isnan(oacore).all():
        CA = ax.contour(oacore, levels, linestyles=':', cmap='coolwarm') #'PiYG')
        fmta = {l: str(l) if l != 0. else r'$\Omega_{a}=0$'for l in levels}
        ax.clabel(CA, levels[::lev], fontsize=11, fmt=fmta, inline=True)
        CA.collections[ctr].set_linewidth(border_width*.8)
        CA.collections[ctr].set_linestyle('-')
    if not np.isnan(obcore).all():
        CB = ax.contour(obcore, levels, linestyles='--', cmap='coolwarm') #BrBG')
        fmtb = {l: str(l) if l != 0. else r'$\Omega_{b}=0$'for l in levels}
        ax.clabel(CB, levels[::lev], fontsize=11, fmt=fmtb, inline=True)
        CB.collections[ctr].set_linewidth(border_width*.8)
        CB.collections[ctr].set_linestyle('-')

    props = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    aborder = get_border(oacore)
    bborder = get_border(obcore)

    if not np.isnan(aborder).all():
        segment = True if x == 'sa' else False
        aint_on = interpolate_border(aborder, segment=segment)
        ax.plot(aint_on[0], aint_on[1], linewidth=border_width, color='grey')
    a_loc = np.mean(np.where(acore > -.1), 1)
    ax.text(a_loc[1], a_loc[0], 'A+', color='darkslategrey', size=12, bbox=props)

    if not np.isnan(bborder).all():
        bint_on = interpolate_border(bborder)
        ax.plot(bint_on[0], bint_on[1], linewidth=border_width, color='grey')
    b_loc = np.mean(np.where(bcore > -.1), 1)
    ax.text(b_loc[1], b_loc[0], 'B+', color='darkslategrey', size=12, bbox=props)

    #REMOVE: hardcoded text for
    #ax.text(.4*n_size, .4*n_size, '0', size=12, color='dimgrey')
    #ax.text(.85*n_size, .85*n_size, '0', size=12, color='dimgrey')
    ax.set_xlim(0, n_size-1)
    ax.set_ylim(n_size-1, 0)

    xticks = np.linspace(0, n_size, n_ticks)
    xticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*xlims, n_ticks)]
    yticks = np.linspace(0, n_size, n_ticks)
    yticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*ylims, n_ticks)]
    ax.axvline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.axhline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(lab_dict[x])
    ax.set_ylabel(lab_dict[y])
    for loc in ['top', 'right', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if fig:
        cbar = fig.colorbar(pos, ax=ax)
        #olabel = r'$\Omega_c$'#=\frac{\rho_{cp}-\rho_{pp}}{\rho_{cp}+\rho_{pp}}$'
        rlabel = r'$r_g$'
        #label = olabel if onion else rlabel
        cbar.set_label(rlabel, size=10)
        cbar.outline.set_visible(False)
        fig.tight_layout()
        name = 'ps_plots_comb/phsp_gcomb_ppr_{}_{}_{}.pdf'.format(x, y, extraname)

        fig.savefig(name)


def plot_miniplots_acore_rewire(fig, ax):
    ylim = ax.get_ylim()
    yrg = ylim[1] - ylim[0]
    xlim = ax.get_xlim()
    xrg = xlim[1] - xlim[0]

    left, bottom, width, height = [0.32, 0.62, 0.175, 0.175]
    axm1 = fig.add_axes([left, bottom, width, height])
    s1 = .625
    sol1 = gfp.rewire_fixed_points(na=.5, c=.9, sa=s1, sb=s1)
    r1a = gfp.cp_correlation_cont(*sol1[2], na=.5, rho=.1, a=.01)
    r1b = gfp.cp_correlation_cont(sol1[2][1], sol1[2][0] , na=.5, rho=.1, a=.01)
    ax.plot(s1, r1a-r1b, 's', color='red')
    #ax.plot([(left+width)*xrg + xlim[0], s1], [(bottom)*yrg - ylim[0], r1], 'grey', alpha=.5)
    plot_mini_net(*sol1[2], ax=axm1)

    left, bottom = [0.13, 0.275]
    axm2 = fig.add_axes([left, bottom, width, height])
    s2 = .15
    sol2 = gfp.rewire_fixed_points(na=.5, c=.5, sa=s2, sb=s2)
    r2a = gfp.cp_correlation_cont(*sol2[0], na=.5, rho=.1, a=.01)
    r2b = gfp.cp_correlation_cont(sol2[0][1], sol2[0][0] , na=.5, rho=.1, a=.01)
    ax.plot(s2, r2a - r2b, 's', color='red')
    plot_mini_net(*sol2[0], ax=axm2)

    left, bottom = [0.32, 0.4]
    axm3 = fig.add_axes([left, bottom, width, height])
    s3 = .95
    sol3 = gfp.rewire_fixed_points(na=.5, c=.5, sa=s3, sb=s3)
    r3a = gfp.cp_correlation_cont(*sol3[0], na=.5, rho=.1, a=.01)
    r3b = gfp.cp_correlation_cont(sol3[0][1], sol3[0][0] , na=.5, rho=.1, a=.01)
    ax.plot(s3, r3a - r3b, 's', color='red')
    plot_mini_net(*sol3[0], ax=axm3)

    left, bottom = [0.32, 0.15]
    axm4 = fig.add_axes([left, bottom, width, height])
    #sol4 = gfp.rewire_fixed_points(na=.5, c=.9, sa=.65, sb=.65)
    r4a = gfp.cp_correlation_cont(*sol1[0], na=.5, rho=.1, a=.01)
    r4b = gfp.cp_correlation_cont(sol1[0][1], sol1[0][0] , na=.5, rho=.1, a=.01)
    ax.plot(s1, r4a - r4b, 's', color='red')
    plot_mini_net(*sol1[0], ax=axm4)

def get_phase_space_grow_combined(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250, extraname=''):
    """
    Data for plot where color is BE's and contour is density's omega
    """
    opath = 'ps_plots_comb/datag_psco_{}{}_{}.p'.format(x, y, extraname)
    rpath = 'ps_plots_comb/datag_pscr_{}{}_{}.p'.format(x, y, extraname)
    if os.path.exists(rpath) and os.path.exists(opath):
        rdata = p.load(open(rpath, 'rb'))
        odata = p.load(open(opath, 'rb'))
        return rdata, odata
    else:
        xvals = np.linspace(*xlims, n_size + 2)[1:-1]
        yvals = np.linspace(*ylims, n_size + 2)[1:-1]
        acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        acore[:], bcore[:] = np.nan, np.nan
        oacore, obcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        oacore[:], obcore[:] = np.nan, np.nan

        for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
            print(i,j)
            params[x] = xv
            params[y] = yv
            dens, ca, cb = gfp.growth_fixed_points_density(**params)
            #Remove alpha borgattis measure
            ca = [x[0] for x in ca]; cb = [x[0] for x in cb]
            acore, bcore, oacore, obcore = classify_densities_comb(dens, ca, cb, acore, bcore, oacore, obcore, i, j, params=params)
        odata = {'acore': oacore, 'bcore': obcore, 'params': params}
        rdata = {'acore': acore, 'bcore': bcore, 'params': params}
        p.dump(odata, open(opath, 'wb'))
        p.dump(rdata, open(rpath, 'wb'))
        return rdata, odata


def get_phase_space_rewire_combined(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250, extraname='', onion=False):
    opath = 'ps_plots_comb/datar_psco_{}{}_{}.p'.format(x, y, extraname)
    path = 'ps_plots_comb/datar_pscr_{}{}_{}.p'.format(x, y, extraname)
    if os.path.exists(path) and os.path.exists(opath):
        data = p.load(open(path, 'rb'))
        odata = p.load(open(opath, 'rb'))
        return data, odata
    else:
        xvals = np.linspace(*xlims, n_size + 2)[1:-1]
        yvals = np.linspace(*ylims, n_size + 2)[1:-1]
        acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        oacore, obcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        st_border = np.zeros((n_size, n_size)); st_border[:] = np.nan
        ost_border = np.zeros((n_size, n_size)); ost_border[:] = np.nan
        acore[:], bcore[:] = np.nan, np.nan
        oacore[:], obcore[:] = np.nan, np.nan
        saved = {}; osaved = {}
        points = {}
        for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
            print(i,j)
            params[x] = xv
            params[y] = yv
            dens, ca, cb = gfp.rewire_fixed_points_density(**params)
            ## remove alphas from borgattis measure
            ca = [x[0] for x in ca]; cb = [x[0] for x in cb]
            points[(i,j)] = [dens, ca, cb]
            acore, bcore, st_border, oacore, obcore, ost_core = classify_densities_rewire_comb(dens, ca, cb, acore, bcore, oacore, obcore, i, j, st_border, ost_border)
        saved['params'] = params
        saved['points'] = points
        saved['acore'] = acore; saved['bcore'] = bcore
        saved['st_border'] = st_border

        osaved['params'] = params
        osaved['points'] = points
        osaved['acore'] = oacore; osaved['bcore'] = obcore
        osaved['st_border'] = ost_border

        p.dump(saved, open(path, 'wb'))
        p.dump(osaved, open(opath, 'wb'))
        return saved, osaved


def get_phase_space_rewire(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250, extraname='', onion=False, alpha=False):
    extraname = f'{extraname}_alpha' if alpha else extraname
    opath = 'ps_plots_onion/datar_psonion_{}{}_{}.p'.format(x, y, extraname)
    path = 'ps_plots_onion/datar_ps_{}{}_{}.p'.format(x, y, extraname)
    dpath = opath if onion else path
    if os.path.exists(dpath):
        data = p.load(open(dpath, 'rb'))
        return data
    else:
        xvals = np.linspace(*xlims, n_size + 2)[1:-1]
        yvals = np.linspace(*ylims, n_size + 2)[1:-1]
        acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        oacore, obcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        st_border = np.zeros((n_size, n_size)); st_border[:] = np.nan
        ost_border = np.zeros((n_size, n_size)); ost_border[:] = np.nan
        acore[:], bcore[:] = np.nan, np.nan
        oacore[:], obcore[:] = np.nan, np.nan
        saved = {}; osaved = {}
        points = {}
        for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
            print(i,j)
            params[x] = xv
            params[y] = yv
            dens, ca, cb = gfp.rewire_fixed_points_density(**params)
            if alpha:
                ca = [x[1] for x in ca]; cb = [x[1] for x in cb]
            else:
                ca = [x[0] for x in ca]; cb = [x[0] for x in cb]
            points[(i,j)] = [dens, ca, cb]
            acore, bcore, st_border = classify_densities_rewire(dens, ca, cb, acore, bcore, i, j, st_border, onion=False)
            oacore, obcore, ost_border = classify_densities_rewire(dens, ca, cb, oacore, obcore, i, j, ost_border, onion=True)
        saved['params'] = params
        saved['points'] = points
        saved['acore'] = acore; saved['bcore'] = bcore
        saved['st_border'] = st_border

        osaved['params'] = params
        osaved['points'] = points
        osaved['acore'] = oacore; osaved['bcore'] = obcore
        osaved['st_border'] = ost_border

        p.dump(saved, open(path, 'wb'))
        p.dump(osaved, open(opath, 'wb'))
        if onion:
            return osaved
        else:
            return saved


def get_sim_data_rewire(x='s', params={'na': .5, 'rho': .015}, xlims=(0, 1), cvals=[.01, .25, .5, .75, .99], x_samp=7, path='', Nnodes=5000, n_samp=10):
    """
    Get simulation data for specified values of c and x (=s, sa, sb)
    """
    if os.path.exists(path):
        sim_data = p.load(open(path, 'rb'))
    else:
        sim_data = {}
        xvals = np.linspace(*xlims, x_samp+2)[1:-1]
        rho = params.pop('rho')
        L = int(rho * Nnodes * (Nnodes-1) / 2)
        i = 0
        for cv, xv in product(cvals, xvals):
            print('{}   ----   c: {} s: {}\n'.format(i, cv, xv))
            if x == 's':
                params['sa'] = xv
                params['sb'] = xv
            else:
                params[x] = xv
            params['c'] = cv

            vals = _get_sim_points(params, N=Nnodes, L=L, n_samp=n_samp)
            for val in vals:
                sim_data[i] = val
                i += 1

            if path:
                fpth = open(path + 'running.p', 'wb')
                p.dump(sim_data, fpth)
                fpth.close()
        if path:
            fpth = open(path, 'wb')
            p.dump(sim_data, fpth)
    return sim_data


def get_phase_space_grow(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250, extraname='', onion=False, alpha=False):
    #NOTE: onion is not entirely implemented, only for development purposes
    # remove/adapt onion on all paths
    opath = 'ps_plots_onion/datag_psonion_{}{}_{}.p'.format(x, y, extraname)
    path = 'ps_plots_onion/datag_ps_{}{}_{}.p'.format(x, y, extraname)
    dpath = opath if onion else path
    print(dpath)
    if os.path.exists(dpath):
        data = p.load(open(dpath, 'rb'))
        return data
    else:
        xvals = np.linspace(*xlims, n_size + 2)[1:-1]
        yvals = np.linspace(*ylims, n_size + 2)[1:-1]
        acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        acore[:], bcore[:] = np.nan, np.nan
        oacore, obcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
        oacore[:], obcore[:] = np.nan, np.nan

        for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
            print(i,j)
            params[x] = xv
            params[y] = yv
            dens, ca, cb = gfp.growth_fixed_points_density(**params)
            if alpha:
                ca = [x[1] for x in ca]; cb = [x[1] for x in cb]
            else:
                ca = [x[0] for x in ca]; cb = [x[0] for x in cb]
            acore, bcore = classify_densities(dens, ca, cb, acore, bcore, i, j, onion=False, params=params)
            oacore, obcore = classify_densities(dens, ca, cb, oacore, obcore, i, j, onion=True, params=params)
        odata = {'acore': oacore, 'bcore': obcore, 'params': params}
        data = {'acore': acore, 'bcore': bcore, 'params': params}
        p.dump(odata, open(opath, 'wb'))
        p.dump(data, open(path, 'wb'))
        if onion:
            return odata
        else:
            return data


def plot_phase_space_grow(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.15}, xlims=(0, 1), ylims=(0, 1), n_size=150, fig=None, ax=None, data=(), title='', n_ticks=11, border_width=8, cmap='mako', extraname='', onion=False):
    """
    Figure 3 subplots. More general case. If onion then plot Omega, else plot Borgatti's r
    """
    if ax is None:
        fig, ax = plt.subplots()
    alph = 'alpha' if alpha else ''
    if not extraname:
        extraname = '_'.join([f'{k}{v}' for k, v in params.items()]) + f'_n{n_size}{alph}'
    data = get_phase_space_grow(x, y, params, xlims, ylims, n_size, extraname=extraname, onion=onion)
    acore, bcore = data['acore'], data['bcore']
    if not onion:
        ax.imshow(acore, vmin=0, vmax=1, cmap=cmap)
        pos = ax.imshow(bcore, vmin=0, vmax=1, cmap=cmap)
    else:
        ax.imshow(acore, vmin=-1, vmax=1, cmap='icefire')
        pos = ax.imshow(bcore, vmin=-1, vmax=1, cmap='icefire')
        levels = np.arange(-1, 1.1, .1)
        CA = ax.contour(acore, levels, cmap='pink')
        CB = ax.contour(bcore, levels, cmap='pink')
        ax.clabel(CA, levels[::2], fontsize=9, fmt='%1.1f') #, inline=True)
        ax.clabel(CB, levels[::2], fontsize=9, fmt='%1.1f') #, inline=True)


    props = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    aborder = get_border(acore)
    bborder = get_border(bcore)

    if not np.isnan(acore).all():
        aint_on = interpolate_border(aborder, segment=True)
        ax.plot(aint_on[0], aint_on[1], linewidth=border_width, color='grey')
        a_loc = np.mean(np.where(acore > -.1), 1)
        ax.text(a_loc[1], a_loc[0], 'A+', color='darkslategrey', size=12, bbox=props)

    if not np.isnan(bcore).all():
        bint_on = interpolate_border(bborder)
        ax.plot(bint_on[0], bint_on[1], linewidth=border_width, color='grey')
        b_loc = np.mean(np.where(bcore > -.1), 1)
        ax.text(b_loc[1], b_loc[0], 'B+', color='darkslategrey', size=12, bbox=props)

    #REMOVE: hardcoded text for
    #ax.text(.4*n_size, .4*n_size, '0', size=12, color='dimgrey')
    #ax.text(.85*n_size, .85*n_size, '0', size=12, color='dimgrey')
    ax.set_xlim(0, n_size-1)
    ax.set_ylim(n_size-1, 0)

    xticks = np.linspace(0, n_size, n_ticks)
    xticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*xlims, n_ticks)]
    yticks = np.linspace(0, n_size, n_ticks)
    yticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*ylims, n_ticks)]
    ax.axvline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.axhline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(lab_dict[x])
    ax.set_ylabel(lab_dict[y])
    for loc in ['top', 'right', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if fig:
        cbar = fig.colorbar(pos, ax=ax)
        olabel = r'$\Omega_c$'#=\frac{\rho_{cp}-\rho_{pp}}{\rho_{cp}+\rho_{pp}}$'
        rlabel = r'$r_c$' if not alpha else r'$\alpha$'
        label = olabel if onion else rlabel
        cbar.set_label(label, size=10)
        cbar.outline.set_visible(False)
        fig.tight_layout()
        if onion:
            name = 'ps_plots_onion/phsp_gonion_{}_{}_{}.pdf'.format(x, y, extraname)
        else:
            name = 'ps_plots_onion/phsp_g_{}_{}_{}.pdf'.format(x, y, extraname)

        fig.savefig(name)


def get_segments(x, y):
    segs = {0: [[x[0], y[0]]], 1: []}
    li = 0
    for (x0, y0), (x1, y1) in zip(zip(x[:-1], y[:-1]), zip(x[1:], y[1:])):
        if (x1-x0 < 2) & (np.abs(y1-y0) < 2):
            segs[li].append([x1, y1])
        else:
            li = np.abs(li-1)
            segs[li].append([x1, y1])
    segs[0] = np.array(segs[0])
    segs[1] = np.array(segs[1])
    return segs


def interpolate_border(border, s=10, invert='', segment=False):
    """
    Interpolate the border of density-dominat regions
    """
    x, y = np.where(border.T == 1)
    if invert == 'x':
        y = [i for _, i in sorted(zip(x, y))]
        x = sorted(x)
    elif invert == 'y':
        x = [i for _, i in sorted(zip(y, x))]
        y = sorted(y)

    if segment:
        segs = get_segments(x, y)
        #### NOTE: this is only for a "circle like case"
        x0 = segs[0][:, 0]
        y0 = segs[0][:, 1]
        x1 = segs[1][:, 0]
        y1 = segs[1][:, 1]
        #x = np.concatenate([x0, np.flip(x1)])
        #y = np.concatenate([y0, np.flip(y1)])
        x = np.concatenate([np.flip(x1), x0])
        y = np.concatenate([np.flip(y1), y0])
    tck, u = interpolate.splprep([x, y], s=s)

    unew = np.arange(0, 1.001, .001)
    out = interpolate.splev(unew, tck)
    return out[0], out[1]



def plot_mini_net(paa, pbb, na=.5, rho=.05, N=500, ax=None, node_size=10):
    """
    Plot a small net with specified Paa and Pbb matrices, of sizes na, with density rho and of N nodes
    """
    import networkx as nx
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2)) #2, 2

    L = rho * N * (N-1) / 2
    Na = int(N * na)
    rho_aa = paa * rho / na**2
    rho_ab = (1-paa-pbb) * rho / (na*(1-na))
    rho_bb = pbb * rho / (1-na)**2
    p = np.array([[rho_aa, rho_ab], [rho_ab, rho_bb]])
    net = nx.stochastic_block_model([Na, N - Na], p=p, seed=10)
    options = {'node_size': node_size, 'alpha': .5, 'ax': ax}

    edges = net.edges()
    net = nx.empty_graph()
    net.add_edges_from(edges)

    pos = nx.spring_layout(net, seed=0) #, iterations=10) #, seed=10)
    nx.draw_networkx_edges(net, pos, edge_color='grey', alpha=.3, ax=ax) #**options)
    anodes = [i for i in range(Na) if i in net]
    bnodes = [i for i in range(Na, N) if i in net]
    nx.draw_networkx_nodes(net, pos, nodelist=anodes, node_shape='s', node_color='orangered', **options)
    nx.draw_networkx_nodes(net, pos, nodelist=bnodes, node_shape='<', node_color='royalblue', **options)
    for lab in ['top', 'bottom', 'left', 'right']:
        ax.spines[lab].set_visible(False)
    return ax


def get_amplification_acore_rewire(n_c=5, n_size=50, xlims=(.5, 1)):
    """
    Data for figure 2, get MFE fixed points classifying stable and unstable solutions
    """
    path = f'amp/acore_c{n_c}_nsize{n_size}_rewire_xlims{xlims[0]}-{xlims[1]}.p'
    if os.path.exists(path):
        data = p.load(open(path, 'rb'))
    else:
        xvals = np.linspace(*xlims, n_size+2)[1:-1]
        if n_c == 5:
            cvals = [0.5, 0.625, 0.75, 0.875, 0.999]
        else:
            cvals = np.linspace(0.5, .999, n_c)
        sngl, stb1, stb2, ustb = np.zeros((n_c, n_size)), np.zeros((n_c, n_size)), np.zeros((n_c, n_size)), np.zeros((n_c, n_size))
        sngl[:], stb1[:], stb2[:], ustb[:] = np.nan, np.nan, np.nan, np.nan
        lt = np.zeros((n_c, n_size))
        params = {'na': .5, 'rho': .1}
        data = {}
        for (j, xv), (i, cv) in product(enumerate(xvals), enumerate(cvals)):
            print(i, j)
            params['c'] = cv
            params['sa'] = xv
            params['sb'] = xv
            dens, ca, cb = gfp.rewire_fixed_points_density(**params)
            sngl, stb1, stb2, ustb = classify_densities_amp_norm(ca, cb, sngl, stb1, stb2, ustb, i, j)
            lt = get_binary_cp_class(lt, dens, i, j)
        data['lt'] = lt
        data['single'] = sngl
        data['stable1'] = stb1
        data['stable2'] = stb2
        data['unstable'] = ustb
        data['xvals'] = xvals
        data['cvals'] = cvals
        p.dump(data, open(path, 'wb'))
    return data


def get_binary_cp_class(lt, dens, i, j):
    """
    lt is one if any group is the core (in terms of ordered densities)
    """
    dens = dens[0]
    if ((dens[0] > dens[1]) and (dens[1] > dens[2])) or ((dens[2] > dens[1]) and (dens[1] > dens[0])):
        lt[i, j] = 1
    return lt


def get_amplification_acore_growth(n_c=6, n_size=50, xlims=(.5, 1)):
    path = f'amp/acore_c{n_c}_nsize{n_size}_growth_xlims{xlims[0]}-{xlims[1]}.p'
    if os.path.exists(path):
        data = p.load(open(path, 'rb'))
    else:
        xvals = np.linspace(*xlims, n_size+2)[1:-1]
        cvals = np.linspace(0.0, 1, n_c)
        sngl = np.zeros((n_c, n_size)); sngl[:] = np.nan
        params = {'na': .5, 'rho': .1, 'sb': .5}
        data = {}
        for (j, xv), (i, cv) in product(enumerate(xvals), enumerate(cvals)):
            print(i, j)
            params['c'] = cv
            params['sa'] = xv
            dens, ca, cb = gfp.growth_fixed_points_density(**params)
            sngl, _, _, _ = classify_densities_amp_acore(ca, sngl, [], [], [], i, j)
        data['single'] = sngl
        data['xvals'] = xvals
        data['cvals'] = cvals
        p.dump(data, open(path, 'wb'))
    return data


def _get_sim_points(params, N=10000, L=1000, n_samp=10, mtype='rewire'):
    t = N*2000
    sim_params = deepcopy(params)
    rho = 2*L / (N*(N-1))
    sim_params.update({'N': N, 'L': L, 't': t, 'n_samp': n_samp})
    na = params['na']
    sim = {}
    if mtype=='rewire':
        sol = gfp.rewire_fixed_points(**params)
        if len(sol) == 3:
            #If there are two stable solutions, we use biased starting points to ensure we get all fixed points
            P0a = (47/50, 1/50)
            P0b = (1/50, 47/50)

            sim_params['P'] = P0a
            ac = gfp.rewire_simul_n(**sim_params)

            sim_params['P'] = P0b
            bc = gfp.rewire_simul_n(**sim_params)

            val = _classify_mean_sim(ac + bc, sol, sim_params)
        else:
            P0 = (1/3, 1/3)
            sim_params['P'] = P0
            sc = gfp.rewire_simul_n(**sim_params)
            val = _classify_mean_sim(sc, sol, sim_params)

    elif mtype=='growth':
        sol = gfp.growth_fixed_points(**params)
        P0 = (1/3, 1/3)
        sim_params['P'] = P0
        sc = gfp.growth_simul_n(**sim_params)
        val = _classify_mean_sim(sc, i, j, stb1, sim_params)

    return val

def _classify_mean_sim(sc, sol, params):
    """
    Get the average simulated fixed points.
    If there are two solutions, classify them according to
    to the closest fixed point.
    Return the mean of the simulations
    """
    if len(sol) == 1:
        val = np.mean(np.array(sc), axis=0)
        sp1 = deepcopy(params)
        sp1['P'] = val
        sim_point = [sp1]
    else:
        val1, val2 = [], []
        for s in sc:
            d1 = (s[0]-sol[0][0])**2 + (s[1]-sol[0][1])**2
            d2 = (s[0]-sol[2][0])**2 + (s[1]-sol[2][1])**2
            if d1 < d2:
                val1.append(s)
            else:
                val2.append(s)
        val1 = np.mean(np.array(val1), axis=0)
        val2 = np.mean(np.array(val2), axis=0)

        sp1 = deepcopy(params)
        sp1['P'] = val1

        sp2 = deepcopy(params)
        sp2['P'] = val2
        sim_point = [sp1, sp2]

    return sim_point


def plot_phase_space_rewire(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(0, 1), ylims=(0, 1), n_size=200, fig=None, ax=None, n_ticks=11, data=(), border_width=9, title='', cbar_ax=None, cmap='mako', segment=True, extraname='', onion=False, alpha=True):
    """
    Old Phase space plots, containing either r, omega or alpha
    """
    if ax is None:
        fig, ax = plt.subplots()
    alph = '_alpha' if alpha else ''
    if not extraname:
        extraname = '_'.join([f'{k}{v}' for k, v in params.items()]) + f'_n{n_size}{alph}'
    if not data:
        data = get_phase_space_rewire(x, y, params, xlims, ylims, n_size, extraname=extraname, onion=onion, alpha=alpha)

    # Plot heatmaps
    acore = data['acore']; bcore = data['bcore']
    center_mask = ~np.isnan(acore + bcore)
    center = np.maximum(acore, bcore) #.5*(acore + bcore)
    nocore = np.nan_to_num(acore) + np.nan_to_num(bcore)
    acore_mask = np.ma.masked_where(center_mask, acore)
    bcore_mask = np.ma.masked_where(center_mask, bcore)
    if not onion:
        ax.imshow(acore_mask, vmin=0, vmax=1, cmap=cmap)
        ax.imshow(bcore_mask, vmin=0, vmax=1, cmap=cmap)
        pos = ax.imshow(center, vmin=0, vmax=1, cmap=cmap)
    else:
        ax.imshow(acore_mask, vmin=-1, vmax=1, cmap='icefire')
        ax.imshow(bcore_mask, vmin=-1, vmax=1, cmap='icefire')
        pos = ax.imshow(center, vmin=-1, vmax=1, cmap='icefire')

    # Plot CP border
    st_border = data['st_border']
    st_border[st_border == 1] = np.nan
    cborder = get_border(st_border)
    if onion:
        aborder_onion = get_border(acore)
        bborder_onion = get_border(bcore)
        acore[acore < 0] = np.nan
        bcore[bcore < 0] = np.nan
    aborder = get_border(acore)
    bborder = get_border(bcore)
    if onion:
        aint_on = interpolate_border(aborder_onion, segment=False)
        ax.plot(aint_on[0], aint_on[1], ':', linewidth=border_width, color='darkgrey')
        bint_on = interpolate_border(bborder_onion, segment=False)
        ax.plot(bint_on[0], bint_on[1], ':', linewidth=border_width, color='darkgrey')

    # Border interpolation might not work for some phase space --
    # -- -- need to adjust settings manually
    aint = interpolate_border(aborder, n_size/6, invert='y', segment=False)
    #bint = interpolate_border(bborder, n_size/6, invert=False, segment=False)
    ax.plot(aint[0], aint[1], linewidth=border_width, color='grey')
    #ax.plot(bint[0], bint[1], linewidth=border_width, color='grey')

    if ~np.isnan(cborder).all():
        segment = True
        cint = interpolate_border(cborder, n_size/4, segment=segment)
        ax.plot(cint[0], cint[1], linewidth=border_width, color='red', linestyle=':')

    # loc doesn't always work if alpha=True (needs some alphas to be )
    a_loc = np.nanmean(np.where(np.nan_to_num(acore_mask.data) > 0), 1)
    b_loc = np.nanmean(np.where(np.nan_to_num(bcore_mask.data) > 0), 1)

    u_loc = np.mean(np.where(nocore == 0), 1)
    props = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    ax.text(a_loc[1], a_loc[0], 'A+', color='darkslategrey', size=12, bbox=props)
    ax.text(b_loc[1], b_loc[0], 'B+', color='darkslategrey', size=12, bbox=props)
    if ~np.isnan(cborder).all():
        c_loc = np.mean(np.where(center > 0), 1)
        ax.text(c_loc[1]-.08*n_size, c_loc[0]-.05*n_size, 'A+ B+\n  U', color='darkslategrey', size=12, bbox=props)
    ax.text(u_loc[1], u_loc[0], '0', size=12, color='darkslategrey', bbox=props)

    ax.axvline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.axhline(n_size/2, linestyle='--', alpha=.7, color='grey')

    xticks = np.linspace(0, n_size, n_ticks)
    xticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*xlims, n_ticks)]
    yticks = np.linspace(0, n_size, n_ticks)
    yticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*ylims, n_ticks)]
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels) #[(xt, xl) for xt, xl in zip(xticks, xticklabels)])
    ax.set_yticklabels(yticklabels) #[(xt, xl) for xt, xl in zip(yticks, xticklabels)])
    ax.set_xlabel(lab_dict[x])
    ax.set_ylabel(lab_dict[y])
    for loc in ['top', 'right', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)

    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if onion:
        name = 'ps_plots_onion/phsp_ronion_{}_{}_{}.pdf'.format(x, y, extraname)
    else:
        name = 'ps_plots_onion/phsp_r_{}_{}_{}.pdf'.format(x, y, extraname)

    if fig:
        cbar = fig.colorbar(pos, ax=ax)
        label = r'$\Omega_c$' if onion else r'$r_c$'
        label = label if not alpha else r'$\alpha$'
        cbar.set_label(label, size=10)
        cbar.outline.set_visible(False)
        #for loc in ['top', 'right', 'left', 'bottom']:
        #    cbar.spines[loc].set_visible(False)
        fig.tight_layout()
        fig.savefig(name)

def plot_abstract():
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(right=.85, wspace=.2)
    cbar_ax = fig.add_axes([.9, .12, .03, .75])
    res_g = p.load(open('plots/data_plot_phasespaceg_sasb.p', 'rb'))
    res_r = p.load(open('plots/data_plot_phasespacer_sasb_rho0.1.p', 'rb'))

    data_g = res_g['acore'], res_g['bcore']
    data_r = res_r['acore'], res_r['bcore']
    params = {'c':.95, 'na':.5, 'rho':.1}
    plot_phase_space_grow(x='sa', y='sb', params=params, xlims=(0,  1), ylims=(0, 1), n_size=200, fig=None, ax=axs[0], data=data_g, title='Growing', n_ticks=11, border_width=5)
    plot_phase_space_rewire(x='sa', y='sb', params=params, xlims=(0, 1), ylims=(0, 1), n_size=200, ax=axs[1], n_ticks=11, data=data_r, border_width=5, title='Rewiring', cbar_ax=cbar_ax)
    axs[0].grid(False)
    #fig.tight_layout()
    fig.savefig('plots/abstract_top.pdf')



def get_border(acore, border=None):
    """
    acore: phase space with nan values where a group is not density-dominant.
    Returns:
        border: phase_space of the same size with ones in transitions from density-dominant to nan
    """
    n_size = acore.shape[0]
    if not border:
        border = np.zeros((n_size, n_size)); border[:] = np.nan
    for i, j in product(range(n_size), range(n_size)):
        if i > 0:
            if np.isnan(acore[i-1, j]) and not np.isnan(acore[i, j]):
                border[i, j] = 1
        if i+1 < n_size:
            if np.isnan(acore[i+1, j]) and not np.isnan(acore[i, j]):
                border[i, j] = 1
        if j > 0:
            if np.isnan(acore[i, j-1]) and not np.isnan(acore[i, j]):
                border[i, j] = 1
        if j+1 < n_size:
            if np.isnan(acore[i, j+1]) and not np.isnan(acore[i, j]):
                border[i, j] = 1
    return border


def classify_densities(dens, ca, cb, acore, bcore, i, j, onion=False, params={}):
    paa, pab, pbb = dens[0]
    if not onion:
        if paa > max([pab, pbb]): #= pab) & (pab >= pbb):
            acore[i, j] = ca[0]
        elif pbb > max([pab, paa]):
            bcore[i, j] = cb[0]
    else:
        na = params.get('na', 0)
        if (paa > max([pab, pbb])):
            acore[i, j] = (pab - pbb)/(pab + pbb)
        elif (pbb > max([paa, pab])):
            bcore[i, j] = (pab - paa)/(pab + paa)
    return acore, bcore


def classify_densities_comb(dens, ca, cb, acore, bcore, oacore, obcore, i, j):
    """
    Classify densities for a single fixed point (instead of 3)
    Returns values for Borgatti's r and omega, both groups may be density-dominant
    """
    paa, pab, pbb = dens[0]

    na = params.get('na', 0)
    if (paa > max([pab, pbb])):
        acore[i, j] = ca[0]
        oacore[i, j] = (pab - pbb)/(pab + pbb)
    elif (pbb > max([paa, pab])):
        bcore[i, j] = cb[0]
        obcore[i, j] = (pab - paa)/(pab + paa)
    return acore, bcore, oacore, obcore


def classify_densities_rewire_comb(dens, ca, cb, acore, bcore, oacore, obcore, i, j, st_border, ost_border):
    """
    Classifiy fixed points for the different values of the rewiring phase space. The classification includes (i) borgatti's r, (ii) omega and (iii) number of solutions for both when a and b are density-dominant

    dens: list of 1 or 3 fixed points (each fixed point consists of 3 group densities)
    ca: borgatti's r correlation when a is core
    cb: borgatti's r correlation when b is core
    acore: plotting values for r correlation when a is core
    bcore: plotting values for r correlation when b is core
    oacore: plotting values for omega when a is core
    obcore: plotting values for omega when b is core
    i, j: indeces for fixed points within phase space
    st_border, ost_border: placeholders to know whether there are 1 or 3 fixed points
    """
    st_border[i, j] = len(dens)
    ost_border[i, j] = len(dens)
    if len(dens) == 1:
        # ONLY ONE FIXED POINT
        acore, bcore, oacore, obcore = classify_densities_comb(dens, ca, cb, acore, bcore, oacore, obcore, i, j)
    elif len(dens) == 3:
        # FOR 3 FIXED POINTS, ONLY THE FIRST AND LAST ARE STABLE
        acore, bcore, oacore, obcore = classify_densities_comb([dens[0]], [ca[0]], [cb[0]], acore, bcore, oacore, obcore, i, j)
        acore, bcore, oacore, obcore = classify_densities_comb([dens[2]], [ca[2]], [cb[2]], acore, bcore, oacore, obcore, i, j)
    return acore, bcore, st_border, oacore, obcore, ost_border

def classify_densities_rewire(dens, ca, cb, acore, bcore, i, j, st_border, onion=False):
    st_border[i, j] = len(dens)
    if len(dens) == 1:
        acore, bcore = classify_densities(dens, ca, cb, acore, bcore, i, j, onion)
    elif len(dens) == 3:
        acore, bcore = classify_densities([dens[0]], [ca[0]], [cb[0]], acore, bcore, i, j, onion)
        acore, bcore = classify_densities([dens[2]], [ca[2]], [cb[2]], acore, bcore, i, j, onion)
    return acore, bcore, st_border

def classify_densities_amp(dens, ca, cb, acore, bcore, i, j, coreval):
    if len(dens) == 1:
        paa, pab, pbb = dens[0]
        acore[i, j] = ca[0]
        bcore[i, j] = cb[0]
        if (paa >= pab) & (pab >= pbb):
            coreval[(i,j)] = 'a'
        elif (pbb >= pab) & (pab >= paa):
            coreval[(i,j)] = 'b'
    elif len(dens) == 3:
        sol_num = 0
        for den, a, b in zip(dens, ca, cb):
            if sol_num != 1:
                paa, pab, pbb = den
                if paa > pbb:
                    acore[i, j] = a
                else:
                    bcore[i, j] = b

                if (paa >= pab) & (pab >= pbb):
                    coreval[(i,j)] = coreval.get((i, j), '') + 'a'
                    acore[i, j] = a

                if (pbb >= pab) & (pab >= paa):
                    coreval[(i,j)] = coreval.get((i, j), '') + 'b'
                    bcore[i, j] = b
            sol_num += 1
    return acore, bcore, coreval



def classify_densities_amp_norm(ca, cb, sngl, stb1, stb2, ustb, i, j):
    """
    Get Borgatti's ra-rb for the different solutions.
    """
    if len(ca) == 1:
        sngl[i, j] = ca[0] - cb[0]
    elif len(ca) == 3:
        stb1[i, j] = ca[0] - cb[0]
        stb2[i, j] = ca[2] - cb[2]
        ustb[i, j] = ca[1] - cb[1]
        if (j > 0): #For visualization of bifurcation
            if (np.isnan(stb1[i, j-1])) & (not np.isnan(stb1[i, j])):
                stb1[i, j-1] = sngl[i, j-1]
            if (np.isnan(stb2[i, j-1])) & (not np.isnan(stb1[i, j])):
                stb2[i, j-1] = sngl[i, j-1]
    return sngl, stb1, stb2, ustb


if __name__ == '__main__':
    import argparse as args
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = args.ArgumentParser()
    parser.add_argument('--mtype', type=str, required=True, default='rewire')
    parser.add_argument('--x', type=str, required=True, default='sa')
    parser.add_argument('--y', type=str, required=True, default='sb')
    parser.add_argument('--c', type=float, default=0.95)
    parser.add_argument('--na', type=float, default=0.5)
    parser.add_argument('--sa', type=float, default=0.75)
    parser.add_argument('--sb', type=float, default=0.75)
    parser.add_argument('--rho', type=float, default=0.075)
    parser.add_argument('--n_size', type=int, default=25)
    parser.add_argument('--onion', type=boolean_string, default=False)


    args = parser.parse_args()
    params = {'sa': args.sa,
            'sb': args.sb,
            'c': args.c,
            'na': args.na,
            'rho': args.rho}
    x = args.x; y = args.y
    extraname=['{}{}'.format(k,v) for k, v in params.items() if k not in [x, y]] + [f'n{args.n_size}']
    if args.mtype == 'rewire':
        plot_phase_space_rewire_combined(x=x, y=y, params=params, n_size=args.n_size, xlims=(0, 1), ylims=(0, 1), extraname='_'.join(extraname)) #, onion=args.onion)
    elif args.mtype == 'grow':
        print(f'onion:{args.onion}')
        plot_phase_space_grow_combined(x=x, y=y, params=params, n_size=args.n_size, xlims=(0, 1), ylims=(0, 1), extraname='_'.join(extraname)) #, onion=args.onion)
    #plot_amplification_rewire(Nnodes=1000, n_sim=5, sim_data=True, n_size=50)
