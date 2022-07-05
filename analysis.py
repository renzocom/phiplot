from phiplot import ces_view, ces_report, network
import pyphi

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import pickle

import logging
import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyphi.models.subsystem import FlatCauseEffectStructure as sep


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Logging config
logging.basicConfig(format='%(asctime)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %I:%M:%S')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)

def compute_ces(subsystem, run_id, save_dir='', rels_max_degree=2):
    '''
    Computes cause-effect structure (distinctions and relations) and save it to a pickle.

    Parameters
    ----------
    subsystem : subsystem to compute CES
    run_id : str or Path
    save_dir : directory to save (ex: Path.cwd())
    rels_max_degree
    '''
    print(subsystem.node_labels)

    # COMPUTE CES
    print('Computing distinctions...')
    distinctions = pyphi.compute.ces(subsystem, parallel=True)
    # distinctions = pyphi.compute.subsystem.ces(subsystem, parallel=True)

    print('Computing relations...')
    # relations = list(pyphi.relations.relations(subsystem, sep(distinctions), max_degree=max_degree))
    relations = list(pyphi.relations.relations(subsystem, sep(distinctions),
                                               max_degree=rels_max_degree,
                                               computation='CONCRETE'))

    print('Saving everything...')
    fname = run_id + '.pkl'
    pickle.dump((subsystem, distinctions, relations), open(save_dir / fname, "wb"))
    print('Done.')

def filter_noncontiguous_distinctions(distinctions, relations):
    '''
    Filter non-contiguous (e.g. ABDE) distinctions and relations.
    '''
    contiguous_distinctions = [d for d in distinctions if ces_view.is_contiguous(d.mechanism)]
    contiguous_mechs = [d.mechanism for d in contiguous_distinctions]

    contiguous_relations = []
    for rel in relations:
        if rel.mechanisms[0] in contiguous_mechs and rel.mechanisms[1] in contiguous_mechs:
            contiguous_relations.append(rel)

    return contiguous_distinctions, contiguous_relations

def filter_conflicting_distinctions(distinctions, relations, rule):
    '''
    Filter conflicts in CES (distinctions over the same purview).

    rule : 'purview_size' or 'small_phi'
    '''

    # FIND CONFLICTS
    node_labels = distinctions[0].node_labels
    ces_conflicts = get_conflicts_in_ces(distinctions)

    n_cause_conflicts, n_effect_conflicts = len(ces_conflicts['cause']), len(ces_conflicts['effect'])
    n_conflicts = n_cause_conflicts + n_effect_conflicts

    logging.debug(ces_report.underline_str(f"PURVIEW CONFLICTS IN CES: {n_conflicts} ({n_cause_conflicts} cause, {n_effect_conflicts} effect)"))

    # RESOLVE CONFLICTS (select winning distinctions)
    logging.debug("C : contiguous, (*) : selected")
    logging.debug("")
    winning_distinctions = {'cause': [], 'effect': []}
    for purview_side in ['cause', 'effect']:
        logging.debug(ces_report.underline_str(purview_side.upper(), '-'))
        for i, (purview, conflict_distinctions) in enumerate(ces_conflicts[purview_side].items()):
            best_distinction = get_best_distinction(conflict_distinctions, 'effect',
                                                    rule) if purview_side == 'cause' else get_best_distinction(conflict_distinctions, 'cause', rule)
            s = ces_view.node_ixs2label(purview, node_labels)
            logging.debug(f"#{i + 1} {s}")

            for distinction in conflict_distinctions:
                s = ces_report.quick_distinction_str(distinction)

                cont = 'C' if ces_view.is_contiguous(distinction.mechanism) else ""
                if best_distinction == distinction:
                    logging.debug(f"{s} {cont} (*)")
                else:
                    logging.debug(f"{s} {cont}")
            logging.debug("")
            winning_distinctions[purview_side].append(best_distinction)

    # FILTER CES
    ces_conflicts_flat = dict()
    ces_conflicts_flat['cause'] = list(itertools.chain.from_iterable(ces_conflicts['cause'].values()))
    ces_conflicts_flat['effect'] = list(itertools.chain.from_iterable(ces_conflicts['effect'].values()))

    losing_distinctions = dict()
    losing_distinctions['cause'] = [d for d in ces_conflicts_flat['cause'] if d not in winning_distinctions['cause']]
    losing_distinctions['effect'] = [d for d in ces_conflicts_flat['effect'] if d not in winning_distinctions['effect']]

    losing_distinctions = list(set(losing_distinctions['cause'] + losing_distinctions['effect']))
    losing_mechanisms = [d.mechanism for d in losing_distinctions]

    filtered_distinctions = [d for d in distinctions if d not in losing_distinctions]

    filtered_relations = []
    for rel in relations:
        if rel.mechanisms[0] in losing_mechanisms or rel.mechanisms[1] in losing_mechanisms:
            pass
        else:
            filtered_relations.append(rel)
    return filtered_distinctions, filtered_relations

def get_conflicts_in_ces(distinctions):
    '''
    Returns conflicts in the cause-effect structure, i.e. distinctions over the same purview.

    Parameters
    ----------
    distinctions : list of distinctions

    Returns
    -------
    dict[purview_side][purview] = list of distinctions
    '''

    def get_purview_label(distinction, cause_or_effect):
        if cause_or_effect == 'cause':
            return distinction.cause_purview
        elif cause_or_effect == 'effect':
            return distinction.effect_purview
        else:
            raise ValuerError("wrong value in cause_or_effect.")

    node_indices = distinctions[0].subsystem.node_indices
    pset = list(pyphi.utils.powerset(node_indices, nonempty=True))

    ces_conflicts = {'cause': dict(), 'effect': dict()}

    for purview_side in ['cause', 'effect']:
        for p in pset:
            conflicts = []
            for m in distinctions:
                if get_purview_label(m, purview_side) == p:
                    conflicts.append(m)
            if len(conflicts) > 1:
                ces_conflicts[purview_side][p] = conflicts

    return ces_conflicts

def get_best_distinction(distinctions, purview_side2check, rule):
    '''
    Function to resolve conflicts in CES (distinctions over the same purview).

    Parameters
    ----------
    distinctions : list of conflicting distinctions
    purview_side2check : 'cause' or 'effect'
    rule : 'purview_size' or 'small_phi'
        'purview_size' : selects mechanism with largest purview, and then with largest small phi.
        'small_phi' : selects mechanism with largest small phi, then with largest purview.

    '''
    if purview_side2check == 'cause':
        get_purview = lambda x: x.cause_purview
    else:
        get_purview = lambda x: x.effect_purview

    if rule == "purview_size":
        purview_lengths = np.array([len(get_purview(distinction)) for distinction in distinctions])
        max_purview_len = np.max(purview_lengths)
        ixs = np.where(max_purview_len == purview_lengths)[0]

        if len(ixs) > 1:
            remaining_distinctions = [distinctions[ix] for ix in ixs]
            phis = [distinction.phi for distinction in remaining_distinctions]
            ix = np.argmax(phis)
            return remaining_distinctions[ix]
        else:
            return distinctions[ixs[0]]

    elif rule == "small_phi":
        phis = [distinction.phi for distinction in distinctions]
        max_phi = np.max(phis)
        ixs = np.where(max_phi == phis)[0]

        if len(ixs) > 1:
            remaining_distinctions = [distinctions[ix] for ix in ixs]
            purview_lengths = [len(get_purview(distinction)) for distinction in remaining_distinctions]
            ix = np.argmax(purview_lengths)
            return remaining_distinctions[ix]
        else:
            return distinctions[ixs[0]]

    else:
        raise ValueError("rule must be either purview_size or small_phi")

def plot_tpm(tpm, figsize=(10, 6), states=None):
    '''
        Plots transition probability matrix as state-by-state and state-by-node.

        Parameters
        ----------
        tpm : array (can be 2d state-by-state, 2d state-by-node or multi dim state-by-node)
    '''

    if tpm.ndim > 2: # convert multidimensional to state-by-state
        tpm_sbn = pyphi.convert.to_2d(tpm)
        tpm_sbs = pyphi.convert.state_by_node2state_by_state(tpm_sbn)
        print(tpm_sbs.shape)
    elif tpm.shape[0]!=tpm.shape[1]: # state-by-node
        tpm_sbn = tpm
        tpm_sbs = pyphi.convert.state_by_node2state_by_state(tpm_sbn)
    else:
        tpm_sbs = tpm
        tpm_sbn = pyphi.convert.state_by_state2state_by_node(tpm_sbs)
        tpm_sbn = pyphi.convert.to_2d(tpm_sbn)

    fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 8]})
    plot_tpm_sbn(tpm_sbn, states=states, ax=axs[0])
    axs[0].set_title('State-by-node TPM')
    plot_tpm_sbs(tpm_sbs, states=states, ax=axs[1])
    axs[1].set_title('State-by-state TPM')

def plot_tpm_sbs(tpm, states=None, ax=None):
    '''
    Plots state-by-state transition probability matrix.

    Parameters
    ----------
    tpm : 2d-array (state, state)
    '''
    assert tpm.shape[0]==tpm.shape[1], "tpm dimensions is invalid."

    if tpm.ndim > 2: # convert multidimensional to state-by-state
        tpm = pyphi.convert.to_2d(tpm)
        tpm = pyphi.convert.state_by_node2state_by_state(tpm)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    n_states = tpm.shape[0]

    ax.imshow(tpm, vmin=0, vmax=1)
    ax.set_xlabel("State")
    ax.set_ylabel("State")
    if states is not None:
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(states, rotation=-90)
        ax.set_yticklabels(states)


def plot_tpm_sbn(tpm, states=None, ax=None):
    '''
    Plots state-by-node transition probability matrix.

    Parameters
    ----------
    tpm : 2d-array (state, node)
    '''
    assert tpm.shape[0]!=tpm.shape[1], "tpm dimensions is invalid."

    if tpm.ndim > 2:
        tpm = pyphi.convert.to_2d(tpm) # convert multidimensional to state-by-node

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 12))

    n_states, n_nodes = tpm.shape
    ax.imshow(tpm, vmin=0, vmax=1)
    ax.set_xticks(range(n_nodes))
    ax.set_xticklabels(np.arange(1, n_nodes + 1))
    ax.set_ylabel("State")
    ax.set_xlabel("Node")
    if states is not None:
        ax.set_yticks(range(n_states))
        ax.set_yticklabels(states)


def plot_network_weights(weights):
    '''
    Plots connectivity matrix, weight matrix and distribution of weights.

    weights : 2d array
        w_ij matrix with weights from node in row i to node in col j.

    '''
    fig, axs = plt.subplots(figsize=(22, 4), nrows=1, ncols=4)

    n_nodes = weights.shape[0]
    cm = weights > 0

    axs[0].imshow(cm, vmin=0, vmax=1)
    axs[0].set_title("Connectivity Matrix")
    axs[0].set_xlabel('Target Node')
    axs[0].set_ylabel('Source Node')

    im = axs[1].imshow(weights, vmin=0, vmax=1)
    axs[1].set_title("Weight Matrix")
    axs[1].set_xlabel('Target Node')
    axs[1].set_ylabel('Source Node')
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    ax = axs[2]
    for n in range(weights.shape[0]):
        ax.plot(range(1, n_nodes + 1), weights[n, :], label=f"{n + 1}", marker='o', markersize=5, color=plt.cm.tab10(n))
        ax.plot(n + 1, weights[n, n], marker='o', markersize=8, color=plt.cm.tab10(n))
    ax.set_xlabel('Target Node')
    ax.legend(title='Source Node', fontsize=8, loc=2)
    ax.set_title("Node Output Weights")

    ax = axs[3]
    for n in range(weights.shape[0]):
        ax.plot(range(1, n_nodes + 1), weights[:, n], label=f"{n + 1}", marker='o', markersize=5, color=plt.cm.tab10(n))
        ax.plot(n + 1, weights[n, n], marker='o', markersize=8, color=plt.cm.tab10(n))
    ax.set_xlabel('Source Node')
    ax.legend(title='Target Node', fontsize=8, loc=2)
    ax.set_title("Node Input Weights")

def preprocess_time_ces(distinctions, relations, conflict_rule='purview_size'):
    '''

    Parameters
    ----------
    distinctions : list of distinctions
    relations : list of relations
    conflict_rule : 'purview_size' or 'small_phi'

    Returns
    -------
    ces : dict[distinction/relation][steps]

    '''

    logging.info(f"raw CES : {len(distinctions)} distinctions, {len(relations)} relations")
    contiguous_distinctions, contiguous_relations = filter_noncontiguous_distinctions(distinctions, relations)
    logging.info(f"contiguous CES : {len(contiguous_distinctions)} distinctions, {len(contiguous_relations)} relations")

    resolved_distinctions, resolved_relations = filter_conflicting_distinctions(distinctions, relations, conflict_rule)
    logging.info(f"resolved CES : {len(resolved_distinctions)} distinctions, {len(resolved_relations)} relations")

    cont_res_distinctions, cont_res_relations = filter_conflicting_distinctions(contiguous_distinctions, contiguous_relations, conflict_rule)
    logging.info(f"contiguous resolved CES : {len(cont_res_distinctions)} distinctions, {len(cont_res_relations)} relations")

    res_cont_distinctions, res_cont_relations = filter_noncontiguous_distinctions(resolved_distinctions, resolved_relations)
    logging.info(f"resolved contiguous CES : {len(res_cont_distinctions)} distinctions, {len(res_cont_relations)} relations")

    ces = {}
    ces['raw'] = {'distinctions':distinctions, 'relations':relations}
    ces['contiguous'] = {'distinctions':contiguous_distinctions, 'relations':contiguous_relations}
    ces['resolved'] = {'distinctions':resolved_distinctions, 'relations':resolved_relations}
    ces['contiguous_resolved'] = {'distinctions':cont_res_distinctions, 'relations':cont_res_relations}
    ces['resolved_contiguous'] = {'distinctions':res_cont_distinctions, 'relations':res_cont_relations}

    return ces

def analyze_time_ces(fpath, save_dir=None, analysis_id='', conflict_rule='purview_size'):
    with open(fpath, "rb") as f:
        print(f"Loading {fpath}")
        subsystem, distinctions, relations = pickle.load(f)

    ces = preprocess_time_ces(distinctions, relations, conflict_rule=conflict_rule)

    # raw_CES = ces_view.CauseEffectStructure(subsystem, mechs=distinctions, rels=relations).get_hasse_pyramid()
    contiguous_CES = ces_view.CauseEffectStructure(subsystem, mechs=ces['contiguous']['distinctions'], rels=ces['contiguous']['relations']).get_hasse_pyramid()
    # resolved_CES = ces_view.CauseEffectStructure(subsystem, mechs=ces['distinctions']['contiguous'], rels=ces['relations']['resolved']).get_hasse_pyramid()
    cont_res_CES = ces_view.CauseEffectStructure(subsystem, mechs=ces['contiguous']['distinctions'], rels=ces['contiguous_resolved']['relations']).get_hasse_pyramid()
    res_cont_CES = ces_view.CauseEffectStructure(subsystem, mechs=ces['contiguous']['distinctions'], rels=ces['resolved_contiguous']['relations']).get_hasse_pyramid()

    CESs = [contiguous_CES, cont_res_CES, res_cont_CES]
    CES_labels = ['ces', 'ces_contiguous_resolved', 'ces_resolved']
    for CES, label in zip(CESs, CES_labels):
        save_path = save_dir / f"{label}_plotly_{analysis_id}.png"
        ces_view.plotly_ces(CES, edge_width=2, show_purview_label=True, title=analysis_id, save_image_path=save_path, show_fig=False);

def analyze_batch_plotly_ces(fnames, load_dir, save_dir, save_name='ces_plotly_subplots', resolve_ces_conflicts=False, conflict_rule='purview_size', n_rows_cols=None):

    def analyze_func(fname, load_dir, fig, subplot):
        fpath = load_dir / fname

        with open(fpath, "rb") as f:
            print(f"Loading {fpath}")
            subsystem, distinctions, relations = pickle.load(f)

        ces = preprocess_time_ces(distinctions, relations, conflict_rule=conflict_rule)

        if resolve_ces_conflicts:
            CES = ces_view.CauseEffectStructure(subsystem, mechs=ces['distinctions']['contiguous'],
                                                rels=ces['relations']['resolved_contiguous']).get_hasse_pyramid()
        else:
            CES = ces_view.CauseEffectStructure(subsystem, mechs=ces['distinctions']['contiguous'],
                                                rels=ces['relations']['contiguous']).get_hasse_pyramid()

        fig = ces_view.plotly_ces(CES, fig=fig,
                                  subplot=subplot,
                                  edge_width=2,
                                  show_purview_label=True,
                                  show_legend=False,
                                  show_fig=False);
        return fig

    n_ces = len(fnames)

    if n_rows_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_ces)))
        n_rows = int(np.ceil(n_ces / n_cols))
    else:
        n_rows, n_cols = n_rows_cols

    layout_width = 1100 * n_cols
    layout_height = 750 * n_rows
    layout = go.Layout(width=layout_width,
                       height=layout_height,
                       paper_bgcolor='#ffffff',
                       plot_bgcolor='#ffffff')
    fig = go.Figure(layout=layout)

    titles = ['.'.join(fname.split('.')[:-1]) for fname in fnames]
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=titles,
                        vertical_spacing=0.02,
                        figure=fig)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    subplots = [(i + 1, j + 1) for i in range(n_rows) for j in range(n_cols)]
    for i, fname in enumerate(fnames):
        print(f'{i+1}/{len(fnames)} : {fname}')
        subplot = subplots[i]
        fig = analyze_func(fname, load_dir, fig, subplot)

    save_image_path = save_dir / f"{save_name}.png"
    fig.write_image(save_image_path, scale=3)

    save_image_path = save_dir / f"{save_name}.svg"
    fig.write_image(save_image_path)

if __name__ == '__main__':

    PROJECT_DIR = Path('/Users/atopos/Drive/science/phi_time/')
    MODEL_DIR = PROJECT_DIR / "models/"
    SAVE_DIR = PROJECT_DIR / "output/tmp"

    # Logging config
    logging.basicConfig(format='%(asctime)s: %(message)s',
                        filename='analysis.log',
                        filemode='w',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %I:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    # CREATE SUBSYSTEM
    n_nodes = 7
    k = 10
    activation = 'logistic'
    scaling = "exponential"
    decay = 1
    normalize_weights = False
    port_in = True
    lc = 0.2
    sc = 1 - lc
    state = [0, 0, 0, 0, 0, 0, 0, 0]

    for n_neighbours in [2]:
        for port_in_weight in [0.7]:

            network_id = "ces_time_network"

            if n_neighbours==1:
                params_id = f"{activation}_k_{k:.0f}_nodes_{n_nodes}_sc_{sc:.2f}_lc_{lc:.2f}_conn_nearest_iit_4.1"
            else:
                # params_id = f"{activation}_k_{k:.0f}_nodes_{n_nodes}_sc_{sc:.2f}_lc_{lc:.2f}_conn_{n_neighbours}_{scaling}_{decay}_iit_4.1"
                # params_id = f"{activation}_k_{k:.0f}_nodes_{n_nodes}_sc_{sc:.2f}_lc_{lc:.2f}_conn_{n_neighbours}_{scaling}_{decay}_port-in_{port_in_weight:.2f}_iit_4.1"
                params_id = f"{activation}_k_{k:.0f}_nodes_{n_nodes}_sc_{sc:.2f}_lc_{lc:.2f}_conn_{n_neighbours}_{scaling}_{decay}_port-in_{port_in_weight:.2f}_input-OFF_iit_4.1"

            run_id = "_".join([network_id, params_id])
            fname = run_id + '.pkl'
            logging.info(f"run id: {params_id}")

            analyze_time_ces(fname, save_dir=SAVE_DIR, analysis_id=params_id)

            D = network.get_time_subsystem(n_nodes=n_nodes,
                                           state=state,
                                           activation=activation,
                                           k=k,
                                           normalize_weights=normalize_weights,
                                           neighbour_scaling=scaling,
                                           n_neighbours=n_neighbours,
                                           sc=sc,
                                           lc=lc,
                                           decay=decay,
                                           port_in=port_in,
                                           port_in_weight=port_in_weight,
                                           full_return=True)

            plot_tpm(D['tpm_state_by_state'])
            plt.suptitle(params_id)
            save_path = SAVE_DIR / f'network_tpm_{params_id}.png'
            plt.savefig(save_path, dpi=200)
            plot_network_weights(D['weights'])
            plt.suptitle(params_id)
            save_path = SAVE_DIR / f'network_weights_{params_id}.png'
            plt.savefig(save_path, dpi=200)