import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import copy
import random


def random_network(N, na, p):
    """
    Create a random network with two groups.
    N: network size
    na: size of group A
    p = 2x2 connection probability between two groups (local densities)
    """
    min_nodes = list(range(int(N * na)))
    Na = len(min_nodes)
    G = nx.stochastic_block_model([Na, N-Na], p)
    return G, Na


def ba_starter(N, fm, h_aa, h_bb):
    """
    Create a base network for growing network.
    Input:
        N: number of nodes
        fm: fraction of nodes of group a (equal to na)
        h_aa: homophily for group a (for distance reference, not implemented)
        h_aa: homophily for group b (for distance reference, not implemented)
    Output:
        G: networkx model
        Na: number of nodes in group a
        dist: distance dictionary
    """
    Na = int(N * fm)
    minority_nodes = range(Na)
    G = nx.Graph()
    node_attribute = {}

    for n in range(N):
        if n < Na:
            G.add_node(n , color = 'red')
            node_attribute[n] = 'minority'
        else:
            G.add_node(n , color = 'blue')
            node_attribute[n] = 'majority'

    dist = defaultdict(int)
    h_ab = 1 - h_aa
    h_ba = 1 - h_bb
    #create homophilic distance ### faster to do it outside loop ###
    for n1 in range(N):
        n1_attr = node_attribute[n1]
        for n2 in range(N):
            n2_attr = node_attribute[n2]
            if n1_attr == n2_attr:
                if n1_attr == 'minority':
                    dist[(n1,n2)] = h_aa
                else:
                    dist[(n1,n2)] = h_bb
            else:
                if n1_attr == 'minority':
                    dist[(n1,n2)] = h_ab
                else:
                    dist[(n1,n2)] = h_ba

    return G, Na, dist


def grow_ba_two(G, sources, target_list, dist, m, cval, ret_counts=False, n_i={}):
    """
    Main iteration for growing model. We pick nodes proportional to degree and accept candidantes with assorative attachment
    Input:
        G: networkx network, obtained from ba_starter
        sources: nodes that have not been connected to the main netowrk
        target_list: nodes that have been connected, potential candidates
        dist: dictionary of distances/assorative attachment. From ba_starter
        m: number of candidate nodes to select
        c: probability for selecting candidates via prefential attachment
        ret_counts: used for fitting maximum-likelihood method.
            If True, returns list containg the number of links created per type of link:
            [AA, AB, AN, BA, BB, BN], where AN/BN mean that a link was not created from a/b.
        n_i: used for fitting maximum-likelihood method
            Network summary statistics at time t (number of nodes per group and degree)
            If not empty, the summary is updated and x_i, the new links, is also returned
    """

    source = np.random.choice(sources)
    _ = sources.remove(source)
    targets = _pick_ba_two_targets_exact(G, source, target_list, dist, m, cval)
    if ret_counts:
        counts = classify_counts(source, targets, G, m)
    if n_i:
        x_i = classify_targets(G, source, targets, Na, n_i)
        n_i = add_target_counts(n_i, x_i) #Note, this is actually n[i+1]
    if targets != set():
        G.add_edges_from(zip([source] * len(targets), targets))
    target_list.append(source)

    ##### SANITY CHECK - compare with n_i
    #na_degs = Counter([G.degree(i) for i in range(Na)])
    #nb_degs = Counter([G.degree(i) for i in range(Na, G.number_of_nodes())])

    if ret_counts:
        return counts
    if n_i:
        return x_i, n_i

def _pick_ba_two_targets_exact(G, source, target_list, dist, m, cval):
    """
    Pick set of new neighbors for node source via growing mdoel.
    Here, a candidate is selected with probability proportional to degree, and
    accepted with probability homophily (or not joined). The process is run m times.
    -----
    G: networkx graph
    source: incoming node for which neighbors are selected
    target_list: list of possible neighbors
    dist: dict of homophilies
    m: number of timess the experiment is run.
    """
    target_list_copy = copy.copy(target_list)
    target_prob_list = []
    for target in target_list_copy:
        tgt_deg = G.degree(target)
        #Small probability added so that we can also pick degree zero nodes
        target_prob =  tgt_deg if tgt_deg > 0 else 0.0001
        target_prob_list.append(target_prob)

    if sum(target_prob_list) == 0:
        return targets
    candidates = set()
    for i in range(min([m, len(target_list_copy)])):
        probs = np.array(target_prob_list) / sum(target_prob_list)
        if random.random() < cval:
            k = np.random.choice(target_list_copy, p=probs, replace=False)
        else:
            k = np.random.choice(target_list_copy, replace=False)
        candidates.add(k)
        prob_idx = target_list_copy.index(k)
        target_list_copy.remove(k)
        target_prob_list.pop(prob_idx)
    # check all candidates and add them
    targets = set()
    for k in candidates:
        if random.random() < dist[(source, k)]:
            targets.add(k)
    return targets

def _rewire_candidates_exact(G, N, Na, c, bias):
    """
    Main iteration for rewiring model. We pick nodes proportional to degree and accept candidantes with assorative attachment. This does not add links, but returns the chosen links.
    Input:
        G: networkx network, obtained from ba_starter
        N: number of nodes
        Na: number of nodes in group a
        c: probability of preferential attachment
        bias: list with assortative atachment parameters
    Output:
        n_link: new link
        d_link: deleted link
        ltype: type of new link (aa, ab, ba, bb)
        typn: type on new node that is rewired (a, b)
        typd: type of node whose link is deleted (a, b)
    """
    startNode = np.random.randint(N)

    if G.degree(startNode) > 0:
        neighs = set(G.neighbors(startNode))
        fset = set(range(N))
        fset = fset.difference(neighs)
        fset = fset.difference(set([startNode]))
        if np.random.random() < c:
            target_prob_list = []
            target_list = []
            for target in fset:
                tgt_pr = G.degree(target)
                target_prob = tgt_pr if tgt_pr > 0 else 0.00001
                target_prob_list.append(target_prob)
                target_list.append(target)
            probs = np.array(target_prob_list) # / sum(target_prob_list)
            endNode = random.choices(target_list, weights=probs)[0]
        else:
            endNode = np.random.choice(list(fset))
        delNode = np.random.choice(list(neighs))
        n_link = (startNode, endNode)
        d_link = (startNode, delNode)
        n_link, d_link, ltype, typn, typd = classify_accept_nodes(n_link, d_link, Na, bias)
    else:
        n_link, d_link, ltype, typn, typd = (), (), None, None, None
    return n_link, d_link, ltype, typn, typd

def classify_accept_nodes(n_link, d_link, Na, bias):
    """
    Rewiring model: accept or reject candidate node based on assorativity parameters.
    d_link is a candidate link for deletion
    """
    accept = False
    startNode, endNode = n_link
    if startNode < Na:
        if endNode < Na:
            typn = 'na'
            ltype = 'aa'
            if np.random.random() < bias[0]:
                accept = True
        else:
            typn = 'nb'
            ltype = 'ab'
            if np.random.random() > bias[0]:
                accept = True
    else:
        if endNode >= Na:
            typn = 'nb'
            ltype = 'bb'
            if np.random.random() < bias[1]:
                accept = True
        else:
            ltype = 'ba'
            typn = 'na'
            if np.random.random() > bias[1]:
                accept = True

    dnode = d_link[1]
    if dnode < Na:
        typd = 'na'
    else:
        typd = 'nb'

    if accept:
        return n_link, d_link, ltype, typn, typd
    else:
        return (startNode,), (), None, None, None

def classify_counts(source, targets, G, m):
    """
    Classify counts for the growing model if the added links are: AA, AB, AN, BA, BB, BN
    Used for validating maximum-likelihood model
    """
    counts = np.zeros(6)
    sr_type = G.nodes[source]['color']
    for target in targets:
        tg_type = G.nodes[target]['color']
        if sr_type == 'red':
            if tg_type == 'red':
                counts[0] += 1
            elif tg_type == 'blue':
                counts[1] += 1
        elif sr_type == 'blue':
            if tg_type == 'red':
                counts[3] += 1
            elif tg_type == 'blue':
                counts[4] += 1
    if sr_type == 'red':
        counts[2] = m - sum(counts[:2])
    elif sr_type == 'blue':
        counts[5] = m - sum(counts[3:5])
    return list(counts)

def classify_targets(G, source, targets, Na, n_i):
    """
    Growing model. Used for computing x_i, the number of added links per link type
    """
    sg = 'a' if source < Na else 'b'
    x_h = {sg + 'a': {}, sg + 'b': {}}
    for tgt in targets:
        tg = 'a' if tgt < Na else 'b'
        tgt_deg = G.degree(tgt)
        x_h[sg + tg][tgt_deg] = x_h[sg + tg].get(tgt_deg, 0) + 1
    x_i = []
    for ltype, cnts in x_h.items():
        for tgt_deg, xk in cnts.items():
            n_k = n_i['n'+ltype[1]].get(tgt_deg, 0)
            xt = [ltype, tgt_deg, n_k, xk]
            x_i.append(xt)
    return x_i

def add_target_counts(n_0, x_i):
    """
    Growing model. Compute networks statistics for fitting maximum likelihood estimates.
    """
    n_i = copy.deepcopy(n_0)

    for x in x_i:
        deg = x[1]
        cnt = x[3]
        tgt = x[0][1]
        nidx = 'n' + tgt

        n_i[nidx][deg] = max([n_i[nidx].get(deg, 0) - cnt, 0])
        n_i[nidx][deg+1] = n_i[nidx].get(deg+1, 0) + cnt

    #tot_deg = sum([sum([cnt for cnt in dc.values()]) for dc in x_i.values()])
    tot_deg = sum([x[3] for x in x_i])
    try:
        sg = x_i[0][0][0]
    except IndexError:
        sg = None
    #sg = 'a' if x_i[0][0]if (x_i['aa'] or x_i['ab']) else 'b' if (x_i['bb'] or x_i['ba']) else None
    if sg:
        n_i['n'+sg][tot_deg] = n_i['n'+sg].get(tot_deg, 0) + 1

    clean_ni = {'na': {}, 'nb': {}}
    for sg in clean_ni:
        clean_ni[sg] = {key: val for key, val in n_i[sg].items() if val > 0}
    return clean_ni

def homophilous_sbm(N, Na, sa, sb, m=2):
    """
    Testing: chose network only with assorative attachment
    """
    Nb = N - Na
    paa = 2 * m * sa / (Na - 1)
    pbb = 2 * m * sb / (Nb - 1)
    pab = m * (Na * (1-sa) + Nb * (1-sb)) / (Na * Nb)
    p = [[paa, pab], [pab, pbb]]
    G, Na = random_network(N, Na / N, p)
    return G, Na

def run_hsbm(N, Na, sa, sb, m=2):
    G, Na = homophilous_sbm(N, Na, sa, sb, m)
    p = get_p(G, Na)
    t = get_t(p)
    corr = measure_core_periph(*t, a=.01)
    return t, corr


def rewire_tc_one(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 1 - close a triangle by choosing a friend of a friend
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            neighbors = list(G.neighbors(startNode))
            newNode = np.random.choice(neighbors)

            neighbors = list(G.neighbors(newNode))
            endNode =  np.random.choice(neighbors)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)

def rewire_tc_two(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 2 - Close a triangle by choosing two neighbors areound a startnode
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 1:
        if np.random.random() < c:
            neighbors = list(G.neighbors(startNode))
            newNodeA, newNodeB = np.random.choice(neighbors, size=2, replace=False)
        else:
            newNodeB = np.random.randint(N)
            newNodeA = startNode
        accept = False
        if (newNodeA != newNodeB) and (not G.has_edge(newNodeA, newNodeB)):
            if newNodeA < Na:
                if newNodeB < Na:
                    if np.random.random() < bias[0]:
                        accept = True
                else:
                    if np.random.random() > bias[0]:
                        accept = True
            else:
                if newNodeB > Na:
                    if np.random.random() < bias[1]:
                        accept = True
                else:
                    if np.random.random() > bias[1]:
                        accept = True

        if accept:
            if remove_neighbor:
                remove_random_neighbor(G, startNode)
            else:
                remove_random_edge(G, edgelist, N_edge)
                edgelist.append([newNodeA, newNodeB])
            G.add_edge(newNodeA, newNodeB)


def rewire_pa_one(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge, return_link=False): #formerly rewire_links
    """
    PA one - create new edge by following a link from a random node
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            newNode = np.random.randint(N)
            while G.degree(newNode) < 1:
                newNode = np.random.randint(N)
            endNode = np.random.choice(list(G.neighbors(newNode)))
        else:
            endNode = np.random.randint(N)

        links = _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge, return_link)
        if return_link:
            return links
    elif return_link:
        return [[], []]


def rewire_network_stepxstep(G, N, Na, c, bias, n_i):
    """
    PA one - create new edge by following a link from a random node
    """
    x_i = {}
    n_i_ = n_i.copy()

    n_link, d_link, ltype, typn, typd = _rewire_candidates_exact(G, N, Na, c, bias)

    if d_link and n_link:
        tgt_deg = G.degree(n_link[1])
        x_i[ltype] = tgt_deg
        n_i[typn][tgt_deg+1] = n_i[typn].get(tgt_deg + 1, 0) + 1
        n_i[typn][tgt_deg] = n_i[typn].get(tgt_deg, 0) - 1 # remove get

        del_deg = G.degree(d_link[1])
        n_i[typd][del_deg] = n_i[typd].get(del_deg, 0) - 1 # remove get
        n_i[typd][del_deg-1] = n_i[typd].get(del_deg - 1, 0) + 1

        n_i_['na'] = {k: v for k, v in n_i['na'].items() if v > 0}
        n_i_['nb'] = {k: v for k, v in n_i['nb'].items() if v > 0}

        G.add_edge(*n_link)
        G.remove_edge(*d_link)

    return x_i, n_i_

def rewire_network_light(G, N, Na, c, bias, n_l, n_i):
    """
    PA one - create new edge by following a link from a random node
    """
    x_i = []
    n_i_ = n_i.copy() #Degree distribution as in rewire_candidates_exact
    n_l = n_l.copy() #Minimum sums required for llik

    n_link, d_link, ltype, typn, typd = _rewire_candidates_exact(G, N, Na, c, bias)

    if d_link and n_link:
        tgt_deg = G.degree(n_link[1])
        tgt_cnt = n_i[typn][tgt_deg]
        x_i = [ltype, tgt_deg, tgt_cnt]
        if typn == 'na':
            n_l[0] += 1
        elif typn == 'nb':
            n_l[1] += 1

        n_i[typn][tgt_deg+1] = n_i[typn].get(tgt_deg + 1, 0) + 1
        n_i[typn][tgt_deg] = n_i[typn].get(tgt_deg, 0) - 1 # remove get

        del_deg = G.degree(d_link[1])
        if typd == 'na':
            n_l[0] -= 1
        elif typd == 'nb':
            n_l[1] -= 1
        #n_l['p' + typd[1]] -= 1
        if del_deg == 1:
            #n_l['u' + typd[1]] -= 1
            if typd == 'na':
                n_l[2] -= 1
            elif typd == 'nb':
                n_l[3] -= 1

        n_i[typd][del_deg] = n_i[typd].get(del_deg, 0) - 1 # remove get
        n_i[typd][del_deg-1] = n_i[typd].get(del_deg - 1, 0) + 1

        n_i_['na'] = {k: v for k, v in n_i['na'].items() if v > 0}
        n_i_['nb'] = {k: v for k, v in n_i['nb'].items() if v > 0}

        G.add_edge(*n_link)
        G.remove_edge(*d_link)

    return x_i, n_l, n_i_


def _rewired_link_data(startNode, endNode, G, Na, bias):
    accept = False
    if (startNode != endNode) and (not G.has_edge(startNode, endNode)):
        if startNode < Na:
            if endNode < Na:
                if np.random.random() < bias[0]:
                    accept = True
            else:
                if np.random.random() > bias[0]:
                    accept = True
        else:
            if endNode >= Na:
                if np.random.random() < bias[1]:
                    accept = True
            else:
                if np.random.random() > bias[1]:
                    accept = True

    if accept:
        lostNode = np.random.choice(list(G.neighbors(startNode)))
        return [(startNode, endNode), (startNode, lostNode)]
    else:
        return []


def _get_candidates_pa_two(G, N):

    newNode = np.random.randint(N)
    while G.degree(newNode) < 1:
        newNode = np.random.randint(N)
    midNode = np.random.choice(list(G.neighbors(newNode)))
    if G.degree(midNode) < 1:
        return _get_candidates_pa_two(G)
    else:
        endNode = np.random.choice(list(G.neighbors(midNode)))
        return endNode


def rewire_pa_two(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    PA one - create new edge by following two links from a random node
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            endNode = _get_candidates_pa_two(G, N)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)

def rewire_tc_three(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 3 - close a triangle by choosing a random neigbor from the set of second neighbors around a link
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 1:
        if np.random.random() < c:
            second_neigh = list()
            for neigh in G.neighbors(startNode):
                second_neigh += list(G.neighbors(neigh))
            second_neigh = np.unique(second_neigh)
            endNode = np.random.choice(second_neigh)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)


def rewire_tc_four(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 4 - close a triangle by choosing a friend of a friend (correcting by the degree of the friends)
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            neighbors = list(G.neighbors(startNode))
            degrees = np.array([1./G.degree(neigh) for neigh in neighbors])
            p = degrees / sum(degrees)
            newNode = np.random.choice(neighbors, p=p)

            neighbors = list(G.neighbors(newNode))
            degrees = np.array([1./G.degree(neigh) for neigh in neighbors])
            p = degrees / sum(degrees)
            endNode =  np.random.choice(neighbors, p=p)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)


def grow_ba_one(G, sources, target_list, dist, m):
    """
    BA 1 - Barabasi-Albert model where we pick nodes propto degree * homophily
    """
    source = np.random.choice(sources)
    _ = sources.remove(source)
    targets = _pick_ba_one_targets(G, source, target_list, dist, m)
    if targets != set():
        G.add_edges_from(zip([source] * m, targets))
    target_list.append(source)


def grow_ba_zero(G, sources, target_list, dist, m):
    """
    BA 0 - Barabasi-Albert model where we pick nodes propto degree and there are groups but no homophily
    """
    source = np.random.choice(sources)
    _ = sources.remove(source)
    targets = _pick_ba_zero_targets(G, source, target_list, dist, m)
    if targets != set():
        G.add_edges_from(zip([source] * m, targets))
    target_list.append(source)



def classify_rewired_links(G, links, Na):
    """
    for each dict of counts of new edges (aa, ab, bb, ba), each
        entry is of the formart deg: count (so count new edges of degree deg)
    """
    x_i = {'aa': {}, 'ab': {}, 'ba': {}, 'bb': {}}
    for (src, tgt) in links:
        sg = 'a' if src < Na else 'b'
        tg = 'a' if tgt < Na else 'b'
        tgt_deg = G.degree(tgt)
        x_i[sg + tg][tgt_deg] = x_i[sg + tg].get(tgt_deg, 0) + 1
    return x_i


def _pick_ba_one_targets(G, source, target_list, dist, m):

    target_prob_dict = {}
    for target in target_list:
        target_prob = (dist[(source,target)]) * (G.degree(target) + 0.00001)
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets
    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets


def _pick_ba_zero_targets(G, source, target_list, dist, m):

    target_prob_dict = {}
    for target in target_list:
        target_prob =  G.degree(target) + 0.00001
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets
    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets


def _pick_ba_two_targets(G, source, target_list, dist, m, cval):
    """
    Pick set of new neighbors for node source via second BA model.
    Here, a candidate is selected with probability proportional to degree, and
    accepted with probability homophily (or not joined). The process is run m times.
    -----
    G: networkx graph
    source: incoming node for which neighbors are selected
    target_list: list of possible neighbors
    dist: dict of homophilies
    m: number of timess the experiment is run.
    """
    target_list_copy = copy.copy(target_list)
    candidates = _choose_random_targets(m, cval, target_list_copy, dist, source)
    m_ba = m - len(candidates)
    target_prob_dict = {}
    for target in target_list_copy:
        tgt_deg = G.degree(target)
        target_prob =  tgt_deg if tgt_deg > 0 else 0.0001
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())
    count_looking = 0
    targets = set()
    if prob_sum == 0:
        return targets
    candidates = set(candidates)
    while len(candidates) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                candidates.add(k)
                break
    # check all candidates and add them
    targets = set()
    for k in candidates:
        if random.random() < dist[(source, k)]:
            targets.add(k)
    return targets



def _choose_random_targets(m, cval, target_list_copy, dist, source):
    m_rand = np.random.binomial(m, 1-cval)
    if len(target_list_copy) >= m_rand:
        candidates = np.random.choice(target_list_copy, m_rand, replace=False)
    else:
        candidates = copy.copy(target_list_copy)
    for k in target_list_copy:
        target_list_copy.remove(k)
    return candidates


def _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge, return_link=False):
    # IF return_link is true, then we don't change the network and get the [l1, l2] set of links, were l1 would be added and l2 removed
    accept = False
    if (startNode != endNode) and (not G.has_edge(startNode, endNode)):
        if startNode < Na:
            if endNode < Na:
                if np.random.random() < bias[0]:
                    accept = True
            else:
                if np.random.random() > bias[0]:
                    accept = True
        else:
            if endNode > Na:
                if np.random.random() < bias[1]:
                    accept = True
            else:
                if np.random.random() > bias[1]:
                    accept = True

    if accept:
        if remove_neighbor:
            rem_edg = remove_random_neighbor(G, startNode)
            #rem_edg = None
        else:
            rem_edg = remove_random_edge(G, edgelist, N_edge)
        if not return_link:
            edgelist.append([startNode, endNode])
            G.add_edge(startNode, endNode)
        else:
        #If we return the selected links, we add the removed edge instead of the new one
            G.add_edge(*rem_edg)
            if edgelist:
                edgelist.append(rem_edg)
        return [startNode, endNode], rem_edg
    return [[], []]


def remove_random_neighbor(G, startNode):
    lostNode = np.random.choice(list(G.neighbors(startNode)))
    G.remove_edge(startNode, lostNode)
    return [startNode, lostNode]


def remove_random_edge(G, edgelist, N_edge):
    #N_edge = len(edgelist)
    edge_idx = np.random.randint(N_edge)
    edge = edgelist.pop(edge_idx)
    G.remove_edge(*edge)
    return edge
    #if len(edgelist) < 1:
    #    for edge in np.random.permutation(G.edges):
    #        edgelist.append(edge)
    #edge = edgelist.pop()
    #G.remove_edge(*edge)


def get_p(G, Na):
    p_aa, p_ab, p_bb = get_l(G, Na)
    n_edges = float(G.number_of_edges())
    p_aa /= n_edges if n_edges > 0 else 1
    p_ab /= n_edges if n_edges > 0 else 1
    p_bb /= n_edges if n_edges > 0 else 1
    return p_aa, p_ab, p_bb

def get_t(p):
    try:
        taa = 2*p[0] / (2*p[0] + p[1])
    except:
        taa = 0
    try:
        tbb = 2*p[2] / (2*p[2] + p[1])
    except:
        tbb = 0
    return taa, tbb

def get_l(G, Na):
    """
    Return total number of links between groups
    """
    l_aa, l_ab, l_bb = 0, 0, 0
    for edge in G.edges():
        if edge[0] >= Na:
            if edge[1] >= Na:
                l_bb += 1
            else:
                l_ab += 1
        else:
            if edge[1] >= Na:
                l_ab += 1
            else:
                l_aa += 1
    return l_aa, l_ab, l_bb

def measure_core_periph(taa, tbb, a=.01):
    """
    Measure core-periphery structure with the T-matrix.
    Where alpha is the prob. of finind a link between groups
    """

    rho_a = np.sqrt((taa - 1/(1+a))**2 + (1-taa + a/(1+a))**2 + (1-tbb - 1)**2 + (tbb)**2)
    rho_b = np.sqrt((tbb - 1/(1+a))**2 + (1-tbb + a/(1+a))**2 + (1-taa - 1)**2 + (taa)**2)

    return rho_a, rho_b

def run_rewiring(N, fm, c, bias, p0, n_iter, track_steps=500, rewire_type="pa_two", remove_neighbor=True, deg_based=False, return_net=False, **kwargs):
    """
    Run the rewiring model with different types of PA
    rewire_type: (str) pa_one, pa_two, tc_one, tc_two, tc_three
    """
    v_types = ["pa_one", "pa_two", "tc_one", "tc_two", "tc_three", "tc_four"]
    assert rewire_type in v_types, "Add valid rewire type"
    rewire_type = 'rewire_' + rewire_type
    rewire_links = eval(rewire_type)
    G, Na = random_network(N, fm, p0)
    edgelist = list(G.edges) if not remove_neighbor else []
    N_edge = len(edgelist)
    P = defaultdict(list)
    if deg_based is False:
        for i in range(n_iter):
            rewire_links(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge)
            if i % track_steps == 0:
                p = get_p(G, Na)
                P['p_aa'].append(p[0])
                P['p_ab'].append(p[1])
                P['p_bb'].append(p[2])
            if i == int(.95 * n_iter):
                p_95 = get_p(G, Na)
                t_95 = get_t(p_95)
        p = get_p(G, Na)
        t = get_t(p)
        converg_d = .5 * (np.abs(t[0] - t_95[0]) + np.abs(t[1] - t_95[1]))
        rho = measure_core_periph(*t)
        return p, t, P, rho, converg_d
    elif deg_based is True:
        x = {}
        n_i = get_ni(G, Na)
        n = {0: n_i}
        #m_dist = kwargs.get('m_dist', ['poisson', 40])
        #m_vals = [1 for _ in range(n_iter)] #_get_m(m_dist, n_iter, 0)
        i = 0
        for _ in range(n_iter):
            x_i, n_i = rewire_network_stepxstep(G, N, Na, c, bias, n_i)
            if x_i:
                x[i] = x_i
                n[i+1] = n_i
                i += 1
        if return_net:
            return G
        else:
            return x, n
    elif deg_based == 'light':
        x = {}
        n_l, deg_dist = light_ni(G, Na)
        n = {0: n_l}
        i = 0
        for _ in range(n_iter):
            x_i, n_l, deg_dist = rewire_network_light(G, N, Na, c, bias, n_l, deg_dist)
            if x_i:
                x[i] = x_i
                n[i+1] = n_l
                i += 1
        if return_net:
            return G
        else:
            return x, n


def get_ni(G, Na):
    n_i = {'na': {}, 'nb': {}}
    for i in range(len(G)):
        deg = G.degree(i)
        if i < Na:
            n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
        else:
            n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
    return n_i

def light_ni(G, Na):
    nt = get_ni(G, Na)
    n_i = [0, 0, 0, 0]
    n_i[0] = sum([n*k for k, n in nt['na'].items()])
    n_i[1] = sum([n*k for k, n in nt['nb'].items()])
    n_i[2] = sum(nt['na'].values())
    n_i[3] = sum(nt['nb'].values())
    return n_i, nt



def run_growing(N, fm, c, bias, p0, n_iter, track_steps=500, rewire_type="ba_two", remove_neighbor=True, m=2, ret_counts=False):
    """
    Run a Barabasi-Albert model for growing a network
    rewire_type: (str) ba_one, ba_two
    WE DONT USE p0, n_iter, remove_neighbor
    WE USE m=2
    """
    #m = 2
    v_types = ["ba_one", "ba_two", "ba_zero"]
    assert rewire_type in v_types, "Add valid rewire type"
    rewire_type = 'grow_' + rewire_type
    grow_links = eval(rewire_type)
    #G, Na = random_network(N, fm, p0)
    h_aa, h_bb = bias
    G, Na, dist = ba_starter(N, fm, h_aa, h_bb)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, m, replace=False))
    counts = []
    for tgt in target_list:
        _ = sources.remove(tgt)
    P = defaultdict(list)
    K = defaultdict(list)
    for i in range(N-m):
        if i % track_steps == 0:
            p = get_l(G, Na) # At some point this has also been get_p
            P['l_aa'].append(p[0])
            P['l_ab'].append(p[1])
            P['l_bb'].append(p[2])
        if ret_counts:
            cnt = grow_links(G, sources, target_list, dist, m, c, ret_counts)
            counts.append(cnt)
        else:
            grow_links(G, sources, target_list, dist, m, c, ret_counts)

    K = None # Instead of counts we used to have K (dict tot deg)
    converg_d = None
    p = get_p(G, Na)
    t = get_t(p)
    rho = measure_core_periph(*t)
    return p, t, (P, counts), rho, converg_d

def run_growing_varying_m(N, fm, c, bias, m_dist=['poisson', 40], m0=10, rewire_type="ba_two", ret_counts=False, deg_based=False):
    """
    Run a Barabasi-Albert model for growing a network
    rewire_type: (str) ba_one, ba_two
    """
    v_types = ["ba_one", "ba_two", "ba_zero"]
    assert rewire_type in v_types, "Add valid rewire type"
    rewire_type = 'grow_' + rewire_type
    grow_links = eval(rewire_type)
    h_aa, h_bb = bias
    G, Na, dist = ba_starter(N, fm, h_aa, h_bb)
    m_vals = _get_m(m_dist, N, m0)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, m0, replace=False))
    counts = []
    for tgt in target_list:
        _ = sources.remove(tgt)
    Paa = list()
    Pbb = list()
    if deg_based is False:
        for i, m in enumerate(m_vals):
            p = get_p(G, Na)
            Paa.append(p[0])
            Pbb.append(p[2])
            cnt = grow_links(G, sources, target_list, dist, m, c, ret_counts=True)
            counts.append(cnt)
        return Paa, Pbb, counts
    else:
        x = {}
        na_src = len([tgt for tgt in target_list if tgt < Na])
        n_i = {'na': {0: na_src}, 'nb': {0: len(target_list)-na_src}}
        n = {0: deg_dist_to_n_light(n_i)}

        for i, m in enumerate(m_vals):
            x_i, n_i = grow_links(G, sources, target_list, dist, m, c, n_i=n_i, Na=Na)
            x[i] = x_i
            n[i+1] = deg_dist_to_n_light(n_i)
        return x, n

def deg_dist_to_n_light(n_i):
    pa = sum([x*k for k, x in n_i['na'].items()])
    pb = sum([x*k for k, x in n_i['nb'].items()])
    ua = sum(n_i['na'].values())
    ub = sum(n_i['nb'].values())

    return [pa, pb, ua, ub]


def _get_m(m_dist, N, m0):
    if isinstance(m_dist, (list, np.ndarray)) and len(m_dist) == N-m0:
        m_vals = m_dist
    elif m_dist[0] == 'poisson':
        m_vals = np.random.poisson(m_dist[1], N - m0)
    cm = []
    for i, m in enumerate(m_vals):
        cm.append(min([m, i + m0]))

    return cm

def total_degree(G, Na):
    Ka, Kb = 0, 0
    for i in range(G.number_of_nodes()):
        if i < Na:
            Ka += G.degree(i)
        else:
            Kb += G.degree(i)
    return Ka, Kb


def theo_random_rewiring(na, s):
    """
    Probability of
    """
    # prob of creating a link
    plink = lambda na, s: na**2*s + 2*(na)*(1-na)*(1-s) + (1-na)**2*s
    paa = na ** 2 * s / plink
    pbb = (1 - na)**2 * s / plink
    pab = 2 * na * (1-na) * (1-s) / plink

    return paa, pab, pbb

