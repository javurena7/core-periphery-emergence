#Data Fitting

## Background

We fit our models to data using a maximum-liklihood approach. 

The main files for fitting are `rewire_degree_fit.py` and `growth_degree_fit.py`. Each file contains objects that fit paramters sa, sb and c via temporal likelihood functions; `RewireFit` and `GrowthFit`, respectively. 

The data must be formatted so that it includes:
1. `x`: observations of the newly created links, in sequence
2. `n`: observations of the network when each link is created, in sequence

## Sample code
```python
    import rewire_degree_fit as rdf
    # Specify links via local densities of smb p0=[[paa, pab], [pab, pbb]], where pij is the probability that two random links in groups i and j are connected. 
    x, n = rdf.get_dataset(sa=.7, sb=.5, na=.3, c=.6, N=1500) #Network size 1500
    RF = RewireFit(x, n)
    sol = RF.solve(x, n) 
    print(sol) #fitted sa, sb, c
```

For `x`, we use a dictionary of timestamps and created links, a link is created from a source to a target (several targets possible for the growing model). In the rewiring model we create one link at a time (so each timestep is formated as `{t:[data_t]}`); in the growing model each time step may have as many links associated in the form of a list of lists (each timestep is formatted as `{t: [[data_t1], [datat2], ...]}`). 
For the growing model, `data_ti` is of the form `[linktype, tgt_k, tgt_cnt, n_cnt]`, where linktype is "ij", with i and j the groups of the source and target nodes, respectively; `tgt_k` is the degree of the target node, `tgt_cnt` is the number of nodes in that group of that degree, and `n_cnt` is the number of target nodes of that degree and group where a link has been created. E.g., `['ba', 34, 14, 2]` means that 2 links were created from a source of group "b" to targets in group "a". Both of those targets were of degree 34, and there were 14 nodes of degree 34 in group "a". 
For the rewiring model, since there can only be one new link at each step, the list is of the form `[linktype, tgt_k, tgt_cnt]`. 

For `n`, we use a dictionary of timestamps and network statistics at time t, `{t: [Pa_tot, Pb_tot, Ua_tot, Ub_tot]}`. We ommit the notation of _t_, but all statistics here are temporal. In our likelihood functions the probability of creating a link from group $a$ to $b$, where $b$ is of degree $k$ is written as:

$$p^{ab}_k = \left(c \frac{n^b_kk}{\sum_j(n^a_jj + n^b_jj)} + (1-c)  \frac{n^b_k}{\sum_j(n^a_j + n^b_j)}\right)s_{ab}$$

Where $n_k$ is the same as `tgt_cnt`, and $k$ is `tgt_k`. The network statistics for n are then we use are then $Pa_{tot}=\sum_j(n^a_j*j)$ and $Ua_{tot} = \sum_j(n^a_j)$.  
