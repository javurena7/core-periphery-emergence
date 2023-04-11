# Phase space analysis

## Background
This folder contains code used for analyzing the phase space of our model. The main files are:

1. `get_fixed_points.py`: used for obtaining the fixed points of both models and measures of core-periphery $r_g$, $\Omega_g$.
2. `plot_fixed_points.py`: used for reproducing Figures 2 and 3. 

## Sample code

To find the fixed point of the growing or rewiring model
```python
  import get_fixed_points as gfp
  eq = gfp.rewire_fixed_points(c=.9, na=.3, sa=.9, sb=.8) 
  #Returns a list of fixed points in terms of the P-matrix of group mixing
```

To plot the phase space of Figure 3 (top right), 
```python
  import plot_fixed_points as pfp
  params={'c': .95, 'na': .5, 'rho': .1}
  pfp.plot_phase_space_rewire_combined('sa', 'sb', params=params, n_size=150) 
  ```
## Disclaimer
Some phase-space plots might need adjusting parameters for correct visualization of borders. 
