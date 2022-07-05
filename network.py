import numpy as np
import pickle
import itertools
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

import pyphi
import pyphi.relations as rel
from pyphi.utils import powerset

# pyphi.config.MEASURE = 'ID' # intrinsic difference
# pyphi.config.REPR_VERBOSITY = 1
# pyphi.config.PARTITION_TYPE = 'TRI'

import string

import numpy as np
import scipy.io

# import visualize_pyphi
from phiplot import network_generator


def get_time_weights(n_nodes, n_neighbours, neighbour_scaling, sc, lc, decay=None, normalize_weights=True):
    '''
    Generate weights_ij (from row i to col j) of time network.

    Parameters
    ----------
    n_nodes: int
    n_neighbours: int (if -1, infinite neighbours)
    scaling: 'constant', 'exponential'
    sc: float
        self-connection
    lc: float
        lateral connection
    decay: float>0


    Returns
    -------
    weights: 2d-array
        connection of node in row i to col j
    '''
    if n_neighbours==-1:
        n_neighbours = n_nodes - 1

    pattern = np.zeros(n_nodes)
    if neighbour_scaling == 'constant':
        pattern[0] = sc
        pattern[1:n_neighbours+1] = lc
    elif neighbour_scaling == 'exponential':  # exponential decay of nearest neighbors
        pattern = np.zeros(n_nodes)
        pattern[0] = sc
        tmp_pattern = [lc * np.exp(- decay * n) for n in range(n_nodes - 1)]
        pattern[1:n_neighbours+1] = tmp_pattern[:n_neighbours]
    else:
        raise ValueError('Invalid scaling method.')

    weights = np.zeros((n_nodes, n_nodes))
    weights[:, 0] = pattern
    for n in range(1, n_nodes):
        tmp_pattern = np.zeros(n_nodes)
        tmp_pattern[n:] = pattern[:-n]
        weights[:, n] = tmp_pattern

    if normalize_weights:  # normalize to input 1
        weights = weights / np.sum(weights, axis=0)

    return weights

def add_port_in(weights, input_weight, normalize=True):
    '''
    Add a port-in node (X) to last node of network (A):
        C <- B <- A <- X

    Parameters
    ----------
    weights
    input_weight :
        0 < float < 1 (if normalize is True)
        float > 0 (if normalize is False)
    normalize

    Returns
    -------

    '''
    n_nodes = weights.shape[0]
    new_weights = np.zeros((n_nodes + 1, n_nodes + 1))
    new_weights[:n_nodes, :n_nodes] = weights
    new_weights[n_nodes, n_nodes - 1] = input_weight
    if normalize: # normalize input to one
        assert input_weight < 1, "input_weights must be < 1 (because normalize is True)"
        new_weights[n_nodes - 1, n_nodes - 1] = 1 - input_weight
    return new_weights

def get_time_subsystem(n_nodes,
                       state=None,
                       activation="logistic",
                       normalize_weights=True,
                       n_neighbours=1,
                       neighbour_scaling='constant',
                       sc=1,
                       lc=None,
                       decay=None,
                       threshold=1/4,
                       exponent=5,
                       l=1,
                       x0=0.5,
                       k=10,
                       port_in=False,
                       port_in_weight=0,
                       full_return=False):
    '''
    Creates time network pyphi Subsystem.

    OBS: Node labels are reversed (e.g. ['D', 'B', 'C', 'A'])

    $$ x_j = \sum_i z_i w_{ji} $$

    Naka Rushton
    ============
    $$ y_j = \frac{(x_j)^{\alpha}}{thr + x_j}  $$

    Logistic Function
    =================
    $$ \frac{l}{l + e^{-k (x - x_0)}} $$


    Parameters
    ----------
    n_nodes : list of 0/1 of size n_nodes

    states : if None, all off is used as state.

    activation : 'logistic' (sigmoid) or 'exponential (Naka-Rushton)'
        'exponential': threshold, exponent
        'logistic': l, x0, k (determinism)

    normalize_weights : bool
        whether to normalize weights input to 1

    scaling :
        'nearest': sc (self-connection), lc (lateral connection)

    full_return : bool

    Returns
    -------
    if full_return:
        dict with variables
    else:
        pyphi.Subsystem

    '''

    node_labels = [string.ascii_uppercase[i] for i in range(n_nodes)][::-1]

    weights = get_time_weights(n_nodes, n_neighbours, neighbour_scaling, sc, lc, decay=decay, normalize_weights=normalize_weights)

    if port_in:
        if port_in_weight==0:
            port_in_weight = lc
        weights = add_port_in(weights, port_in_weight, normalize=normalize_weights)
        node_labels = node_labels + ['X']
        n_nodes = n_nodes + 1

    if activation == "exponential":
        tpm, cm = get_nakarushton_time_tpm(weights.T, threshold=threshold, exponent=exponent)  # back to old convention
        network = pyphi.Network(tpm, cm=cm, node_labels=node_labels)

    elif activation == "logistic":

        network = network_generator.get_net(mech_func=['l'] * n_nodes, weights=weights, l=l, x0=x0, k=k,
                                                            node_labels=node_labels)

    else:
        raise ValueError("'activation' must be 'exponential' or 'logistic'.")

    if state is None:
        state = [0] * n_nodes  # all off

    subsystem = pyphi.Subsystem(network, state, nodes=list(range(n_nodes - 1)))

    if full_return:
        tpm_state_by_node = pyphi.convert.to_2d(network.tpm)
        tpm_state_by_state = pyphi.convert.state_by_node2state_by_state(tpm_state_by_node)
        return {'subsystem': subsystem,
                'weights': weights,
                'tpm_state_by_node': tpm_state_by_node,
                'tpm_state_by_state': tpm_state_by_state,
                'cm': network.cm}
    else:
        return subsystem

def get_nakarushton_time_tpm(weights, threshold=1/4, exponent=5):
    '''
    y_i = sum_j x_j * w_ij

    z_i = y_i ** exponent / (threshold + y_i)

    Parameters
    ----------
    weights: 2d-array (n_nodes, n_nodes)
        connection of node in col j to row i (it is counter-intuitive)
        TODO: receive transposed weights and iterate rows (now columns)
    threshold: float
    exponent: float

    Returns
    -------
    state-by-node TPM : 2d-array (n_states, n_nodes)

    '''
    #THE RETURN OF NAKA-RUSHTON
    n_nodes = len(weights)

    pset = pyphi.utils.powerset(np.arange(n_nodes))

    tpm = np.zeros([2 ** n_nodes, n_nodes])

    for inds in pset:
        istate = np.zeros(n_nodes)
        for y in range(0,len(inds)):
            istate[inds[y]] = 1

        sw = np.zeros(n_nodes,dtype='f')
        sw_norm = np.zeros(n_nodes)
        for z in range(0,len(weights)):
            inputs = istate

            sw[z] = sum(inputs * weights[z]) ** exponent

            sw_norm[z] = sw[z]/(threshold + sw[z])

        V = 0;
        for v in range(0,n_nodes):
            V = V + istate[v]*2**v
        tpm[int(V)] = tuple(sw_norm)

    cm = np.zeros(shape=weights.shape)
    for x in range(0,cm.shape[0]):
        for y in range(0,cm.shape[1]):
            cm[x,y] = int(abs(weights[y,x])>0)

    return tpm, cm



def compute_ks_relations(subsystem, mechanisms, ks):
    all_purviews = pyphi.relations.separate_ces(mechanisms)
    ks_relations = []
    for k in ks:
        relata = [pyphi.relations.Relata(subsystem, pair) for pair in itertools.combinations(all_purviews, k)]
        k_relations = [pyphi.relations.relation(relatum) for relatum in relata]
        k_relations = list(filter(lambda r: r.phi > 0, k_relations))
        ks_relations.extend(k_relations)
    return ks_relations

def compute_k_relations_chunk(chunk):
    relata = chunk
    k_relations = [pyphi.relations.relation(relatum) for relatum in relata]
    k_relations = list(filter(lambda r: r.phi > 0, k_relations))    
    return k_relations

def compute_parallel_ks_relations(subsystem, mechanisms, ks, n_jobs=-1, chunk_size=5000):
    all_purviews = pyphi.relations.separate_ces(mechanisms)
    ks_relations = []
    for k in ks:
        relata = [pyphi.relations.Relata(subsystem, pair) for pair in itertools.combinations(all_purviews, k)]
        chunks = chunk_iterable(relata, chunk_size)
        k_relations = Parallel(n_jobs=n_jobs, verbose=10, backend='multiprocessing')(
                delayed(compute_k_relations_chunk)(chunk) for chunk in tqdm(chunks)
            )
        k_relations_flat = list(itertools.chain.from_iterable(k_relations))
        ks_relations.extend(k_relations_flat)
    return ks_relations

def chunk_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

# def calc_ces_from_subsystem(subsystem, ks, fpath=None, parallel_ces=True, parallel_rels=False, n_jobs=-1, ):

#     print('Computing distinctions...')
#     ces = pyphi.compute.subsystem.ces(subsystem, parallel=parallel_ces)

#     print('Computing relations...')
#     ks = [2]
#     rels = compute_ks_relations(subsystem, ces, ks)

#     print('Saving...')

#     pickle.dump((ces, rels), open(fpath, "wb"))
#     print('Done.')

# def getNRTimeTPM(threshold=1/4,lc=1/4, sc=1, exponent=5,units=8):
#     #THE RETURN OF NAKA-RUSHTON (OLD)
#     weights = np.array([[  sc,   lc,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
#                         [ 0.0,   sc,   lc,  0.0,  0.0,  0.0,  0.0,  0.0],
#                         [ 0.0,  0.0,   sc,   lc,  0.0,  0.0,  0.0,  0.0],
#                         [ 0.0,  0.0,  0.0,   sc,   lc,  0.0,  0.0,  0.0],
#                         [ 0.0,  0.0,  0.0,  0.0,   sc,   lc,  0.0,  0.0],
#                         [ 0.0,  0.0,  0.0,  0.0,  0.0,   sc,   lc,  0.0],
#                         [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   sc,   lc],
#                         [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   sc]])
#     weights = weights[0:units,0:units]

#     nN = len(weights)

#     pset = powerset(np.arange(nN))

#     newtpm = np.zeros([2**nN,nN])

#     for inds in pset:
#         istate = np.zeros(nN)
#         for y in range(0,len(inds)):
#             istate[inds[y]] = 1

#         sw = np.zeros(nN,dtype='f')
#         swN = np.zeros(nN)
#         for z in range(0,len(weights)):
#             inpt = istate

#             sw[z] = sum(inpt*weights[z])**exponent

#             swN[z] = sw[z]/(threshold + sw[z])

#         V = 0;
#         for v in range(0,nN):
#             V = V + istate[v]*2**v
#         newtpm[int(V)] = tuple(swN)

#     cm = np.zeros(shape=weights.shape)
#     for x in range(0,cm.shape[0]):
#         for y in range(0,cm.shape[1]):
#             cm[x,y] = int(abs(weights[y,x])>0)

#     return newtpm, cm

    
if __name__ == '__main__':
    assert pyphi.config.REPERTOIRE_DISTANCE == 'IIT_4.0_SMALL_PHI'
    assert pyphi.config.IIT_VERSION == 'maximal-state-first'
    assert pyphi.config.PARTITION_TYPE == 'TRI'

    project_dir = Path("/Users/atopos/Drive/research/time")

    sc = 0.25
    lc = 1.00
    n_nodes = 10

    run_id = "time_logistic_ces_rels_nodes_8_sc_{:.2f}_lc_{:.2f}_k_10_iit_4.0".format(sc, lc)
    fpath = (project_dir / 'output' / (run_id + '.p'))
    # if not fpath.is_file():
    print(f"run id: {run_id}")

    subsystem = get_time_subsystem(n_nodes=8, scaling='nearest', sc=0.25, lc=1., activation="logistic", k=1)

    print('Computing distinctions...')
    ces = pyphi.compute.subsystem.ces(subsystem, parallel=True)

    print('Computing relations...')
    ks = [2]
    rels = compute_ks_relations(subsystem, ces, ks)

    print('Saving...')

    pickle.dump((ces, rels), open(fpath, "wb"))
    print('Done.')