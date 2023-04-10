import matplotlib.pyplot as plt; plt.ion()
import sys
sys.path.append('..')
import get_fixed_points as gfp
import numpy as np

lsts = {0: '-',
        1: '-',
        2: '-',
        3: '--',
        4: '--'}
xlabs = {'sa':r'$s_a$',
        'sb':r'$s_b$',
        'c':r'$c$',
        'na':r'$n_a$'}

def plot_fullrange_policy_rewire(x='sa', params={'c':0.95, 'na':.1, 'sb':.5}, rho=0.1, obs_0={}, obs_n={}, pols=[],title='', fig=None, ax=None, xlims=(0, 1), data=None, nsize=150, legend=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5,4))

    if not data:
        data = get_pred_vals_rewire(x, params, xlims, rho, nsize)
    alims, blims, ilims = get_core_limits(data)
    for lim in alims:
        ax.fill_betweenx([-1, 1], *lim, color='orangered', alpha=.2)
    for lim in blims:
        ax.fill_betweenx([-1, 1], *lim, color='royalblue', alpha=.2)
    for lim in ilims:
        ax.fill_betweenx([-1, 1], *lim, color='magenta', alpha=.2)

    for i, line in enumerate(data):
        if np.any(line) and i < 3:
            xs = line[:, 0]
            lpaa = r'$P_{aa}^*$' if i == 0 else ''
            lpbb = r'$P_{bb}^*$' if i == 0 else ''
            ax.plot(xs, line[:, 1], lsts[i], color='orangered', label=lpaa)#, alpha=.6)
            ax.plot(xs, line[:, 2], lsts[i], color='royalblue', label=lpbb)#, alpha=.6)
        #if np.any(line) and i == 3:
        #    xs = line[:, 0]
        #    ax.plot(xs, line[:, 1], lsts[i], color='maroon', label=r'$r^*_a$' )
        #if np.any(line) and i == 4:
        #    xs = line[:, 0]
        #    ax.plot(xs, line[:, 1], lsts[i], color='darkblue', label=r'$r^*_b$')
    ax.set_xlim(xlims)
    if (obs_n) and (obs_0):
        plot_observed(x, obs_0, obs_n, ax)
    if (obs_0) and (pols):
        plot_policies(x, obs_0, pols, ax)
    ylims = (-.05, 1)
    ax.set_ylim(ylims)
    ax.set_ylabel(r'$P_{aa}$' + ', '+ r'$P_{bb}$')
    ax.set_xlabel(xlabs[x])
    if legend:
       ax.legend(loc='upper left')
    #fig.tight_layout()
    #fig.savefig('plots/policy_sample_2.pdf')
    return data

def plot_fullrange_policy_grow(x='sa', params={'c':0.95, 'na':.1, 'sb':.5}, rho=0.1, obs_0={}, obs_n={}, pols=[], title='', fig=None, ax=None, xlims=(0, 1), data=None, nsize=25, legend=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5,4))

    if not data:
        data = get_pred_vals_grow(x, params, xlims, rho, nsize)
    alims, blims, ilims = get_core_limits(data)
    for lim in alims:
        ax.fill_betweenx([-1, 1], *lim, color='orangered', alpha=.2)
    for lim in blims:
        ax.fill_betweenx([-1, 1], *lim, color='royalblue', alpha=.2)
    for lim in ilims:
        ax.fill_betweenx([-1, 1], *lim, color='magenta', alpha=.2)

    for i, line in enumerate(data):
        if np.any(line) and i < 3:
            xs = line[:, 0]
            lpaa = r'$P_{aa}^*$' if i == 0 else ''
            lpbb = r'$P_{bb}^*$' if i == 0 else ''
            ax.plot(xs, line[:, 1], lsts[i], color='orangered', label=lpaa)#, alpha=.6)
            ax.plot(xs, line[:, 2], lsts[i], color='royalblue', label=lpbb)#, alpha=.6)
        if np.any(line) and i == 3:
            xs = line[:, 0]
            ax.plot(xs, line[:, 1], lsts[i], color='maroon', label=r'$r^*_a$' )
        if np.any(line) and i == 4:
            xs = line[:, 0]
            ax.plot(xs, line[:, 1], lsts[i], color='darkblue', label=r'$r^*_b$')
    ax.set_xlim(xlims)
    if (obs_n) and (obs_0):
        plot_observed(x, obs_0, obs_n, ax)
    if (obs_0) and (pols):
        plot_policies(x, obs_0, pols, ax)
    ylims = (-.05, 1)
    ax.set_ylim(ylims)
    ax.set_ylabel(r'$P_{aa}$' + ', '+ r'$P_{bb}$' + ', ' +r'$r$')
    ax.set_xlabel(xlabs[x])
    if legend:
       ax.legend(loc='upper left')
    #fig.tight_layout()
    #fig.savefig('plots/policy_sample_2.pdf')
    return data

def plot_observed(x, obs0, obsn, ax):
    pa, pb = obs0.get('paa'), obs0.get('pbb')
    pan, pbn = obsn.get('paa'), obsn.get('pbb')

    bbox = dict(boxstyle='round', facecolor='royalblue', alpha=0.3)
    rbox = dict(boxstyle='round', facecolor='orangered', alpha=0.3)

    xv = obs0.get(x)
    ax.axvline(xv, ls=':', alpha=.5, color='grey')
    ax.text(xv, -0.03, xlabs[x] + r'$^0$', alpha=.9, color='grey')
    ax.axhline(pa, ls=':', alpha=.5, color='orangered')
    ax.text(.9, pa+.02, r'$P_{aa}^0$', alpha=1, color='grey', bbox=rbox)
    ax.axhline(pb, ls=':', alpha=.5, color='royalblue')
    ax.text(.9, pb+.02, r'$P_{bb}^0$', alpha=1, color='grey', bbox=bbox)
    ax.quiver(xv, pa, 0, pan-pa, units='y', scale=1, alpha=.8, color='orangered')
    ax.quiver(xv, pb, 0, pbn-pb, units='y', scale=1, alpha=.8, color='royalblue')
    ax.plot(xv, pa, 'o', color='orangered')
    ax.plot(xv, pan, 'X', color='orangered')
    ax.text(xv+.02, pan, r'$P_{aa}^n$', alpha=1, color='grey', bbox=rbox)
    ax.plot(xv, pb, 'o', color='royalblue')
    ax.plot(xv, pbn, 'X', color='royalblue')
    ax.text(xv+.02, pbn, r'$P_{bb}^n$', alpha=1, color='grey', bbox=bbox)


def plot_policies(x, obs0, pols, ax):
    ### pols = list of format [xv, [(pa0, pb0), (pa1, pb1), ..., (pan, pbn)]]
    pa, pb = obs0.get('paa'), obs0.get('pbb')
    for pol in pols:
        pa0, pb0 = pa, pb
        for yr in pol[1]:
            ax.quiver(pol[0], pa0, 0, yr[0]-pa0, units='y', scale=1, alpha=.5, color='orangered')
            ax.quiver(pol[0], pb0, 0, yr[1]-pb0, units='y', scale=1, alpha=.5, color='royalblue')
            pa0, pb0 = yr[0], yr[1]

def get_pred_vals_rewire(x, params, lims, rho, nsize=100):
    xvals = np.linspace(*lims, nsize+2)[1:-1]
    preds = [[], [], [], [], []] #unique, topstable, lowstable, r_acore, r_bcore
    for xv in xvals:
        params[x] = xv
        na = params['na']
        sols = gfp.rewire_fixed_points(**params)
        classify_rewire_sols(sols, na, rho, preds, xv)

    preds = [np.array(pred) for pred in preds]
    return preds

def get_pred_vals_grow(x, params, lims, rho, nsize=100):
    xvals = np.linspace(*lims, nsize+2)[1:-1]
    preds = [[], [], [], [], []] #unique, topstable, lowstable, r_acore, r_bcore
    for xv in xvals:
        params[x] = xv
        na = params['na']
        sols = gfp.growth_fixed_points(**params)
        classify_rewire_sols(sols, na, rho, preds, xv)

    #preds[1].insert(0, preds[0][-1])
    #preds[2].insert(0, preds[0][-1])
    preds = [np.array(pred) for pred in preds]
    return preds


def classify_rewire_sols(sols, na, rho, preds, xv):
    #import pdb; pdb.set_trace()
    if len(sols) == 1:
        rx, grp, core_status = check_core(sols[0], na, rho)
        #unique
        preds[0].append([xv, sols[0][0], sols[0][1]])
        if grp == 'a':
            preds[3].append([xv, rx, core_status])
        elif grp == 'b':
            preds[4].append([xv, rx, core_status])

    elif len(sols) == 3:
        preds[1].append([xv, sols[0][0], sols[0][1]])
        preds[2].append([xv, sols[2][0], sols[2][1]])
        for i in [0, 2]:
            rx, grp, core_status = check_core(sols[i], na, rho)
            if grp == 'a':
                preds[3].append([xv, rx, core_status])
            elif grp == 'b':
                preds[4].append([xv, rx, core_status])

def check_core(sol, na, rho):
    ra = gfp.cp_correlation_cont(sol[0], sol[1], na, rho, 0.01)
    rb = gfp.cp_correlation_cont(sol[1], sol[0], 1-na, rho, 0.01)

    if (ra > 0) and (ra > rb):
        rhoa, rhoab, rhob = get_local_densities(rho, na, sol[0], sol[1])
        cnd = 1 if (rhoa > rhoab) and (rhoab > rhob) else 0
        return ra, 'a', cnd
    elif (rb > 0) and (rb > ra):
        rhoa, rhoab, rhob = get_local_densities(rho, na, sol[0], sol[1])
        cnd = 1 if (rhob > rhoab) and (rhoab > rhoa) else 0
        return rb, 'b', cnd
    else:
        return 0, '', 0

def get_local_densities(rho, na, paa, pbb):
    rho_aa = paa * rho / na**2
    rho_bb = pbb * rho / (1-na)**2
    rho_ab = (1-paa-pbb) * rho / (na*(1-na)) #TODO check if correct
    #print(rho_aa, rho_ab, rho_bb)
    return rho_aa, rho_ab, rho_bb


def get_core_limits(data):
    ## NOTE: this assumes that there can only be one intersection (acore and bcore are two continious ranges)
    acore = data[3]
    bcore = data[4]
    acore_lims = _core_limits(acore) if np.any(acore) else []
    bcore_lims = _core_limits(bcore) if np.any(bcore) else []
    #TODO: add case where there no core or only one core
    if (not acore_lims) and (not bcore_lims):
        return [], [], []
    elif (acore_lims) and (not bcore_lims):
        return [acore_lims], [], []
    elif (bcore_lims) and (not acore_lims):
        return [], [bcore_lims], []
    elif acore_lims[0] < bcore_lims[0]:
        alims, blims, ilims = get_intersection(acore_lims, bcore_lims)
        return alims, blims, ilims
    elif bcore_lims[0] < acore_lims[0]:
        blims, alims, ilims = get_intersection(bcore_lims, acore_lims)
        return alims, blims, ilims
    else:
        return [], [], []


def get_intersection(acore_lims, bcore_lims):
    alims, blims, ilims = [], [], []
    if acore_lims[0] < bcore_lims[0]:
        if acore_lims[1] < bcore_lims[0]:
            alims = [acore_lims]
            blims = [bcore_lims]
        else:
            alims = [[acore_lims[0], bcore_lims[0]]]
            if acore_lims[1] < bcore_lims[1]:
                ilims = [[bcore_lims[0], acore_lims[1]]]
                blims = [[acore_lims[1], bcore_lims[1]]]
            else:
                ilims = [bcore_lims]
                alims += [[bcore_lims[1], acore_lims[1]]]

    return alims, blims, ilims

def _core_limits(acore):
    acore_x = []
    curr = acore[0, 2]
    if acore[0, 2] == 1:
        acore_x.append(acore[0, 0])
    for i in range(1, acore.shape[0]):
        if acore[i, 2] != curr:
            acore_x.append(acore[i, 0])
            curr = acore[i, 2]
    if curr == 1:
        acore_x.append(acore[-1, 0]) #acore[i, 0])
#    if acore[-1, 0] != last_x:
#        acore_x.append()
    return acore_x

#MAIN FUNC for geting policy arrows
def get_policies(num, t_size, r_steps, params, x, P0, vals=[.1 ,-.1, .2, -.2, .5, -.5], mtype='rewire'):
    t = 2 * num * r_steps / t_size
    params['t'] = np.linspace(0, t, max([int(t_size*10), 5000]))
    params['P'] = P0
    xv = params.pop(x)
    xvpols = np.array([xv+i for i in vals])
    xvpols = xvpols[(xvpols > 0) & (xvpols < 1)]
    pols = []
    for xvp in xvpols:
        params[x] = xvp
        steps = _policy_steps(num, params, mtype)
        pol = [xvp, steps]
        pols.append(pol)
    return pols

def _policy_steps(num, params, mtype):
    if mtype == 'rewire':
        ysol = gfp.rewire_path(**params)
    elif mtype == 'grow':
        ysol = gfp.grow_path(**params)
        ysol = ysol[:, :2]
    steps = _split_last_vals(ysol, num)
    return steps

def _split_last_vals(a, n):
    k, m = divmod(len(a), n)
    steps = []
    for i in range(n):
        st = a[(i+1)*k+min(i+1, m)-1]
        steps.append(st)
    return steps
