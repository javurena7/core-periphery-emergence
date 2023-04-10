import plot_fixed_points as pfp

### sa VS sb

params={'c': .95, 'na': .5, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'sb', params=params, n_size=150, title='Equal groups, high PA', segment=True)

params={'c': .95, 'na': .1, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'sb', params=params, n_size=150, title='Unequal groups, high PA', segment=True)

params={'c': .05, 'na': .5, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'sb', params=params, n_size=150, title='Equal groups, low PA')

params={'c': .05, 'na': .1, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'sb', params=params, n_size=150, title='Unequal groups, low PA')

params={'c': .5, 'na': .5, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'sb', params=params, n_size=150, title='Equal groups, medium PA')

params={'c': .5, 'na': .1, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'sb', params=params, n_size=150, title='Unequal groups, medium PA')

#### sa VS c

params={'sb': .75, 'na': .5, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'c', params=params, n_size=150, title='Equal groups, ' + r'$s_b=0.75$', segment=False)

params={'sb': .75, 'na': .1, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'c', params=params, n_size=150, title='Unequal groups, ' + r'$s_b=0.75$', segment=False)

params={'sb': .5, 'na': .5, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'c', params=params, n_size=150, title='Equal groups, ' + r'$s_b=0.5$')

params={'sb': .5, 'na': .1, 'rho': .15}
pfp.plot_phase_space_rewire('sa', 'c', params=params, n_size=150, title='Unequal groups, ' + r'$s_b=0.5$')

### na VS c

params={'sa': .75, 'sb': .75, 'rho': .15}
pfp.plot_phase_space_rewire('na', 'c', params=params, n_size=150, title=r'$s_a=s_b=0.75$', segment=False)

# no borders
#params={'sa': .5, 'sb': .5, 'rho': .15}
#pfp.plot_phase_space_rewire('na', 'c', params=params, n_size=150, title=r'$s_a=s_b=0.75$')

# no borders
#params={'sa': .5, 'sb': .75, 'rho': .15}
#pfp.plot_phase_space_rewire('na', 'c', params=params, n_size=150, title=r'$s_a=0.5, s_b=0.75$')
