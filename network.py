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

import numpy as np
import scipy.io

def get_nakarushton_time_tpm(weights, threshold=1/4, exponent=5):
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

def get_time_weights(scaling, sc, lc=None, decay=None, zero_cross=None, n_nodes=8):
    
    if scaling == 'single':
        pattern = np.zeros(n_nodes)
        pattern[0], pattern[1] = sc, lc
    
    elif scaling == 'linear': # nearest neighbors
        k = sc/zero_cross
        pat = [(zero_cross - n) * k for n in range(zero_cross)]
        pattern = np.zeros(n_nodes)
        pattern[:zero_cross] = pat
            
    elif scaling == 'exponential':
        pattern = np.zeros(n_nodes)
        pattern[0] = sc
        pattern[1:] = [lc * np.exp(- decay * n) for n in range(n_nodes-1)]
    else:
        raise ValueError('Invalid scaling method.')
    
    weights = np.zeros((n_nodes, n_nodes))
    weights[0, :] = pattern
    for n in range(1, n_nodes):
        tmp = np.zeros(n_nodes)
        tmp[n:] = pattern[:-n]
        weights[n, :] = tmp

    return weights

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

    project_dir = Path("/Users/atopos/Drive/research/time")

    # sc_vec = [0.00, 0.25, 0.5, 0.75, 1.00]
    # lc_vec = [0.00, 0.25, 0.5, 0.75, 1.00]
#     for sc in sc_vec:
#         for lc in lc_vec:
    sc = 0.25
    lc = 1.00
    n_nodes = 10

    run_id = "time_ces_rels_nodes_10_sc_{:.2f}_lc_{:.2f}_iit_4.0".format(sc, lc)
    fpath = (project_dir/'output'/(run_id+'.p'))
    # if not fpath.is_file():
    print(f"run id: {run_id}")
    # labels = ['A','B','C','D','E','F','G','H']
    labels = ['J', 'I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
    state_alloff = [0] * n_nodes
#             tpm_time, cm_time = getNRTimeTPM(sc=sc, lc=lc)
    weights = get_time_weights('single', sc=sc, lc=lc, n_nodes=n_nodes)
    tpm, cm = get_nakarushton_time_tpm(weights, threshold=1/4, exponent=5)
    network_time = pyphi.Network(tpm, cm=cm, node_labels=labels)
    subsystem_time = pyphi.Subsystem(network_time, state_alloff)

    print('Computing distinctions...')
    ces_time = pyphi.compute.subsystem.ces(subsystem_time, parallel=True)

    print('Computing relations...')
    ks = [2]
    rels_time = compute_ks_relations(subsystem_time, ces_time, ks)

    print('Saving...')

    pickle.dump((ces_time, rels_time), open(fpath, "wb"))
    print('Done.')

