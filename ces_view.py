#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/06/2021
@author: renzocom
"""

import itertools
import numpy as np

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import collections

from phiplot import ces_report

MAGENTA = '#e264ed'
LIGHT_BLUE = '#66f0ff'
YELLOW = '#f7d631'
LIGHT_GRAY = "#d6d6d6"

class CauseEffectStructure():
    '''
    Class for cause-effect structures.

    CauseEffectStructure stores the distinctions and relations in a graph where the nodes are
    mechanisms and cause/effect purviews and the edges relations between purviews and connections
    between mechanisms and its purviews.

    Parameters
    ----------
    ces : list of mechanisms (pyphi.models.mechanism.Concept)
    rels : list of 2-relations (pyphi.relations.Relation)
    graph : None (or nx.Graph())
    subsystem : None (or list of str)

    TODO figure out why nx.draw(CES) is different from nx.draw(CES._CES)
    '''
    def __init__(self, subsystem, mechs=None, rels=None, graph=None):
        self.subsystem = subsystem
        if 'X' in subsystem.node_labels:
            self.node_labels = list(subsystem.node_labels)[:-1]
        else:
            self.node_labels = list(subsystem.node_labels)

        self.n_nodes = len(self.node_labels)

        if graph is not None:
            self._CES = graph
        else:
            self._CES = nx.Graph()
        if mechs is not None:
            if self.node_labels!=list(mechs[0].node_labels):
                print("WARNING: node_labels from subsystem do not match node_labels from mechanisms")
            for mech in mechs:
                self.add_mechanism(mech)
        if rels is not None:
            for rel in rels:
                self.add_relation(rel)

        self.nodes = self._get_nodes()
        self.edges = self._get_edges()
        self.adj = self._get_adj()
        self.subgraph = self._CES.subgraph
        self.mechs = self._get_mechs()
        self.purviews = self._get_purviews()
        self.mech_edges = self._get_mech_edges()
        self.rel_edges = self._get_rel_edges()

        self.is_multigraph = self._CES.is_multigraph
        self.is_directed = self._CES.is_directed

        self.mech_nodes = self._get_mech_nodes()

        self.add_hasse_layout(warp=0)

    def __next__(self):
        return self._CES.__next__()

    def __iter__(self):
        return self._CES.__iter__()

    def _get_nodes(self):
        return self._CES.nodes

    def _get_edges(self):
        return self._CES.edges

    def _get_adj(self):
        return self._CES.adj

    def _get_mech_nodes(self):
        return [s for s in self.nodes if len(s.split('_')) == 2]

    def _get_mechs(self):
        return [node for node in self.nodes if (len(node.split('_')) == 2) and (node.split('_')[0] == 'm')]

    def _get_purviews(self):
        return [node for node in self.nodes if (len(node.split('_')) == 4) and (node.split('_')[2] in ['c', 'e'])]

    def _get_mech_edges(self):
        # TODO improve with is_mech, is_purview, etc functions
        return [edge for edge in self.edges if (len(edge[0].split('_'))==2) or len(edge[1].split('_'))==2]

    def _get_rel_edges(self):
        return [edge for edge in self.edges if (len(edge[0].split('_'))!=2) and len(edge[1].split('_'))!=2]

    def add_mechanism(self, mech):
        mech_label, cause_label, effect_label = get_mech_label(mech), get_mech_label(mech, 'cause'), get_mech_label(
            mech, 'effect')
        mech_cause_label, mech_effect_label = '_'.join([mech_label, cause_label]), '_'.join([mech_label, effect_label])

        self._CES.add_node(mech_label, **{'type': 'MECH',
                                          'color': 'blue',
                                          'ixs': mech.mechanism,
                                          'phi': mech.phi,
                                          'name': mech_label.split('_')[1]})

        self._CES.add_node(mech_cause_label, **{'type': 'CAUSE',
                                                'color': 'red',
                                                'ixs': mech.cause.purview,
                                                'phi': mech.cause.phi,
                                                'name': cause_label.split('_')[1]})

        self._CES.add_node(mech_effect_label, **{'type': 'EFFECT',
                                                 'color': 'green',
                                                 'ixs': mech.effect.purview,
                                                 'phi': mech.effect.phi,
                                                 'name': effect_label.split('_')[1]})

        self._CES.add_edge(mech_label, mech_cause_label, **{'type': 'MECH_CAUSE',
                                                            'rel_type_color': 'black',
                                                            'set_rel_color': 'black'})
        self._CES.add_edge(mech_label, mech_effect_label, **{'type': 'MECH_EFFECT',
                                                             'rel_type_color': 'black',
                                                             'set_rel_color': 'black'})

    def add_relation(self, rel):
        typeA, typeB = rel.relata[0].direction.name, rel.relata[1].direction.name
        mech_A_label, mech_B_label = get_rel_label(rel, 'mechs', self.node_labels)
        purview_A_label, purview_B_label = get_rel_label(rel, 'purviews', self.node_labels)

        mech_purview_A_label, mech_purview_B_label = '_'.join([mech_A_label, purview_A_label]), '_'.join(
            [mech_B_label, purview_B_label])

        rel_tmp_A, rel_tmp_B = get_rel_label(rel, 'purviews',  self.node_labels, add_prefix=False)
        set_rel, set_rel2 = set_relation(rel_tmp_A, rel_tmp_B)

        rel_type = f'{typeA}_{typeB}'

        type2color = {'CAUSE_CAUSE': 'red', 'CAUSE_EFFECT': 'pink', 'EFFECT_CAUSE': 'pink', 'EFFECT_EFFECT': 'green'}

        set_rel2color = {'inclusion': MAGENTA, 'connection': LIGHT_BLUE,
                         'equivalence': YELLOW}  # magenta / light blue / yellowish

        self._CES.add_edge(mech_purview_A_label, mech_purview_B_label, **{'type': rel_type,
                                                                          'mech_A': mech_A_label,
                                                                          'mech_B': mech_B_label,
                                                                          'purview_A': purview_A_label,
                                                                          'purview_B': purview_B_label,
                                                                          'rel_type': set_rel,
                                                                          'rel_type2': set_rel2,
                                                                          'phi': rel.phi,
                                                                          'rel_type_color': type2color[rel_type],
                                                                          'set_rel_color': set_rel2color[set_rel2]})

    def substructure(self, mechs, mode):
        '''
        Returns cause-effect substructure from selected mechanisms.

        Parameters
        ----------
        Returns sub-CES of a set of mechanisms and its relations.
        mechs : list of str with mechanism labels
        mode : 'interior' (relations between 'mechs') or 'exterior' (all relations from 'mechs')
        '''

        if mode == 'interior':
            return self.substructure_interior(mechs)
        elif mode == 'exterior':
            return self.substructure_exterior(mechs)
        else:
            raise KeyError(f"{mode} is not a valid 'mode' (interior or exterior).")

    def substructure_interior(self, mechs):
        mechs = ['m_' + mech for mech in mechs]
        purviews = [list(self.adj[mech]) for mech in mechs]
        purviews = [y for x in purviews for y in x]

        subgraph = self.subgraph(mechs + purviews)

        return CauseEffectStructure(self.subsystem, graph=subgraph)

    def substructure_exterior(self, mechs):
        '''TODO: remove edges which are not related to mechs.'''

        mechs = ['m_' + mech for mech in mechs]
        purviews = [list(self.adj[mech]) for mech in mechs]
        purviews = [y for x in purviews for y in x]

        related_purviews = [list(self.adj[purview]) for purview in purviews]
        related_purviews = [p for pp in related_purviews for p in pp if len(p.split('_')) == 4]
        related_purviews = list(set(purviews + related_purviews))

        related_mechs = ['_'.join(p.split('_')[:2]) for p in related_purviews]
        related_mechs = list(set(related_mechs))

        subgraph = self.subgraph(related_mechs + related_purviews)

        return CauseEffectStructure(self.subsystem, graph=subgraph)

    def substructure_composition_of_interiors(self, shapes):
        '''
        Returns the cause effect structure of the composition between the substructure interior of each shape (list of mechanisms).
        e.g. the CES of the interior of ['AB', 'BC', 'ABC'] with the interior of ['DE', 'EF'].

        Parameters
        ----------
        CES : CauseEffectStructure()
        shapes : list of list of mech labels

        Returns
        -------
        new CES
        '''
        new_ces = nx.Graph()
        for mechs in shapes:
            shape_ces = self.substructure_interior(mechs).to_graph()
            new_ces = nx.compose(new_ces, shape_ces)

        return CauseEffectStructure(self.subsystem, graph=new_ces)

    def to_graph(self):
        ''' Returns CES as graph (networkx.classes.graph.Graph).'''
        return self._CES

    def add_hasse_layout(self, warp=0, triangle_base=1, warp_mode='exponential', purview_vertical_offset=0.2, purview_horizontal_offset=0.1):
        '''
        Adds position layout to CES graph based on Hasse diagram of contiguous mechanisms.

        Parameters
        ----------
        triangle_base : float
        purview_vertical_offset : float
        purview_horizontal_offset : float
        '''

        node2pos = hasse_layout(n_elements=self.n_nodes, warp=warp, triangle_base=triangle_base, warp_mode=warp_mode)

        mech2pos = {'m_' + ''.join([self.node_labels[ix] for ix in sets]): pos for sets, pos in node2pos.items()}

        purview2pos = {}
        for mech in self.mechs:
            purviews = list(self._CES.adj[mech])

            for purview in purviews:
                purview_type = purview.split('_')[2]

                mech_pos = mech2pos[mech]

                if purview_type == 'c':
                    purview2pos[purview] = np.array(
                        [mech_pos[0] + purview_horizontal_offset, mech_pos[1] - purview_vertical_offset])

                else:  # purview_type=='e'
                    purview2pos[purview] = np.array(
                        [mech_pos[0] + purview_horizontal_offset, mech_pos[1] + purview_vertical_offset])

        pos = {**mech2pos, **purview2pos}

        nx.set_node_attributes(self._CES, pos, 'pos')

    def get_hasse_pyramid(self):
        '''
        Returns cause-effect Hasse pyramid structure.

        Parameters
        ----------
        triangle_base : float
        purview_vertical_offset : float
        purview_horizontal_offset :float

        Returns
        -------
        CauseEffectStructure

        '''
        triangles = hasse_triangles(n_elements=self.n_nodes)
        triangle_labels = [nodes_ixs2label(triangle, self.node_labels) for triangle in triangles]
        hasse_ces = self.substructure_composition_of_interiors(triangle_labels)
        return hasse_ces

    def get_reduced_ces(self, restrict_to_hasse_pyramid=False):
        '''
        Returns reduced CES, where nodes are mechanisms and edges are the 'relation square'
        (relations between all causes and effects between two mechanisms).

        Parameters
        ----------
        hasse_pyramid : bool

        Returns
        -------
        nx.DiGraph object

        '''
        reduced_CES = ReducedCauseEffectStructure(self)
        if restrict_to_hasse_pyramid:
            reduced_CES.restrict_to_hasse_pyramid()
        return reduced_CES

class ReducedCauseEffectStructure():
    '''
    Class  for reduced cause-effect structure, where relations between pairs of mechanism
    is represented as a single edge.

    Stores CES as a networkx graph, where nodes are mechanisms and edges are the 'relation square'
    (relations between all causes and effects between two mechanisms).

    Parameters
    ----------
    CES : CauseEffectStructure object
    '''

    def __init__(self, CES=None, n_nodes=None, node_labels=None, red_CES_graph=None):
        '''
        CES : CauseEffectStructure object
        n_nodes : int
        node_labels : list of strings
        red_CES_graph : nx.Graph
        '''

        if CES is not None:
            self.n_nodes = CES.n_nodes
            self.node_labels = CES.node_labels
            self.graph = self.reduce_CES(CES)
        else:
            self.n_nodes = n_nodes
            self.node_labels = node_labels
            self.graph = red_CES_graph

    def copy(self):
        return ReducedCauseEffectStructure(n_nodes=self.n_nodes,
                                           node_labels=self.node_labels,
                                           red_CES_graph=self.graph.copy())

    

    def reduce_CES(self, CES):
        '''
        Returns nx.DiGraph() with reduced CES graph.

        Parameters
        ----------
        CES : CauseEffectStructure object.
        '''

        red_CES = nx.DiGraph()

        # add nodes
        for mech in CES.mech_nodes:
            red_CES.add_node(mech, **CES.nodes[mech])

        # add edges
        for u_mech in CES.mech_nodes:
            # cause and effect purview
            mech_purviews = list(CES.adj[u_mech])

            # connected purviews
            edges_of_purviews = [list(CES.adj[purview]) for purview in mech_purviews]

            # connected mechanisms
            connected_mechs = list(
                set(['_'.join(edge.split('_')[:2]) for edges in edges_of_purviews for edge in edges if
                     len(edge.split('_')) == 4]))

            for v_mech in connected_mechs:

                # TODO this is sensitive to the order of mech_nodes (low to high order, left to right),
                #  and assures there are no bidirectional edges

                if (u_mech != v_mech) and ((u_mech, v_mech) not in red_CES.edges) and (
                        (v_mech, u_mech) not in red_CES.edges):
                    red_CES.add_edge(u_mech, v_mech)

        # add edge (relations) information

        for mech_edge in red_CES.edges:

            u_mech, v_mech = mech_edge

            # walks to mechanism's purviews
            u_mech_purviews = list(CES.adj[u_mech])
            v_mech_purviews = list(CES.adj[v_mech])

            u_cause_purview, u_effect_purview = (u_mech_purviews[0], u_mech_purviews[1]) if \
                u_mech_purviews[0].split('_')[2] == 'c' else (u_mech_purviews[1], u_mech_purviews[0])
            v_cause_purview, v_effect_purview = (v_mech_purviews[0], v_mech_purviews[1]) if \
                v_mech_purviews[0].split('_')[2] == 'c' else (v_mech_purviews[1], v_mech_purviews[0])

            rel_square = dict(u_cause2effect_u={},
                              v_cause2effect_v={},
                              u_cause2effect_v={},
                              u_effect2cause_v={},
                              u_cause2cause_v={},
                              u_effect2effect_v={})

            for edge, key in zip([(u_cause_purview, u_effect_purview),
                                  (v_cause_purview, v_effect_purview),
                                  (u_cause_purview, v_effect_purview),
                                  (u_effect_purview, v_cause_purview),
                                  (u_cause_purview, v_cause_purview),
                                  (u_effect_purview, v_effect_purview)], rel_square.keys()):

                # relation exists
                if edge in CES.edges:
                    info = CES.edges[edge]
                    # check if mech order of info in graph edge matches with order of new edge
                    edge_A = info['mech_A'] + '_' + info['purview_A']
                    edge_B = info['mech_B'] + '_' + info['purview_B']
                    if edge_A==edge[0] and edge_B==edge[1]:
                        rel_square[key] = info['rel_type']
                    elif edge_A==edge[1] and edge_B==edge[0]:
                        rel_square[key] = flip_rel_type(info['rel_type'])
                    else:
                        raise ValueError(f"Inconsistent information: {edge_A, edge_B, edge}")

                else:
                    rel_square[key] = ' '


            pattern = '_'.join(rel_square.values())
            # pattern = '_'.join([d['rel_type'] for _, d in rel_square.items()])
            # pattern2 = '_'.join([d['rel_type2'] for _, d in rel_square.items()])
            # pattern3 is insensitive to symmetry of relation square
            # (e.g. '<_=_>_<_=_<' and '=_<_>_<_=_>' are equivalent)
            pattern3 = frozenset([pattern, flip_rel_square(pattern)]) 
            
            red_CES.edges[mech_edge]['pattern'] = pattern
            red_CES.edges[mech_edge]['pattern3'] = pattern3
            
            red_CES.edges[mech_edge]['causal_flow'] = self.calc_causal_flow(u_cause_purview, u_effect_purview, v_cause_purview, v_effect_purview)
            red_CES.edges[mech_edge]['causal_flow_norm'] = self.calc_causal_flow(u_cause_purview, u_effect_purview,
                                                                            v_cause_purview, v_effect_purview, norm=True)

        return red_CES

    def calc_causal_flow(self, u_cause_purview, u_effect_purview, v_cause_purview, v_effect_purview, norm=False):
        '''
        Computed causal flow from u to v.

        Parameters
        ----------
        Input are of the form 'm_ABC_c_BCD'
        '''
        u_c = set(u_cause_purview.split('_')[-1])
        u_e = set(u_effect_purview.split('_')[-1])
        v_c = set(v_cause_purview.split('_')[-1])
        v_e = set(v_effect_purview.split('_')[-1])

        if norm:
            return (len(u_e.intersection(v_c)) / len(v_c)) - (len(v_e.intersection(u_c)) / len(u_c))
        else:
            return len(u_e.intersection(v_c)) - len(v_e.intersection(u_c))

    def restrict_to_hasse_pyramid(self):
        '''
        Restrict relations to the ones in the hasse pyramid.
        '''

        triangles = hasse_triangles(n_elements=self.n_nodes)
        triangle_labels = [nodes_ixs2label(triangle, self.node_labels) for triangle in triangles]

        sub_mech_CES = nx.DiGraph()
        for triangle in triangle_labels:
            mechs = ['m_' + mech for mech in triangle]
            subgraph = self.graph.subgraph(mechs)
            sub_mech_CES = nx.compose(sub_mech_CES, subgraph)

        self.graph = sub_mech_CES

    def get_colormap_from_relation_pattern(self, pattern_field='pattern', given_pattern2color=None, colormap=['Set1', 'Set2'], max_n_colors=17):
        '''
        Returns relation square pattern to color map based on most common patterns.

        Parameters
        ----------
        pattern_field: name of attribute of graph edge with pattern.
        '''

        sorted_patterns, _ = self.get_sorted_patterns(pattern_field)

        if given_pattern2color is not None:
            sorted_patterns = [p for p in sorted_patters if p not in given_pattern2color.keys()]
            exclude_rgb_colors = list(given_pattern2color.values())
        else:
            exclude_rgb_colors = None

        n_colors = max_n_colors if len(sorted_patterns) >= max_n_colors else len(sorted_patterns)
        colors = color_array(n_colors, len(sorted_patterns) - n_colors, colormap=colormap, exclude_rgb_colors=exclude_rgb_colors)

        new_pattern2color = {pattern: color for pattern, color in zip(sorted_patterns, colors)}


        return new_pattern2color

    def color_edges(self, pattern2color, pattern_field='pattern', color_field='pattern_color', min_black=0, max_white=0.8):
        '''
        Add color field to edges according colormap.

        min_black : between 0 and 1 (0 is black)
        max_white : between 0 and 1 (1 is white)
        '''
        edge2pattern = nx.get_edge_attributes(self.graph, pattern_field)

        uncolored_patterns = list(set(edge2pattern.values()) - set(pattern2color.keys()))
        n_uncolored_patterns = len(uncolored_patterns)
        if n_uncolored_patterns > 0:
            print("Incomplete pattern2color map: completing missing patterns with gray tones.")
            grays = get_gray_colors(n_uncolored_patterns, min_black=min_black, max_white=max_white)
            pattern2color_extra = {pattern : gray for pattern, gray in zip(uncolored_patterns, grays)}
            pattern2color = {**pattern2color, **pattern2color_extra}

        edge2color = {edge: pattern2color[pattern] for edge, pattern in edge2pattern.items()}
        nx.set_edge_attributes(self.graph, edge2color, color_field)

    def get_edge_colors(self, color_field='pattern_color'):
        return nx.get_edge_attributes(self.graph, color_field)

    def color_edges_by_relation_pattern(self, pattern_field='pattern', color_field='pattern_color',
                                       colormap=['Set1', 'Set2'], max_n_colors=12):
        '''
        Add 'color_field' to edges according edge patterns from 'pattern_field'.
        '''
        pattern2color = self.get_colormap_from_relation_pattern(pattern_field=pattern_field, colormap=colormap, max_n_colors=max_n_colors)
        self.color_edges(pattern2color, pattern_field=pattern_field, color_field=color_field)

    def get_mechanisms_in_triangle(self, reference_mechanism, side_length, full=True):
        '''
        Returns mechanisms in triangle starting from reference_point (bottom left vertex) in the Hasse diagram.

        Example
        -------
        >>> red_CES.get_mechanisms_in_triangle('m_IHGFE', 2, full=True)
        ['m_IHGFE', 'm_HGFED', 'm_GFEDC', 'm_IHGFED', 'm_HGFEDC', 'm_IHGFEDC']
        >>> red_CES.get_mechanisms_in_triangle('m_IHGFE', 2, full=False)
        ['m_IHGFE', 'm_GFEDC', 'm_IHGFEDC']
        '''
        if 'm' in reference_mechanism:
            reference_mechanism = reference_mechanism.split('_')[1]

        reference_indices = tuple(self.node_labels.index(element) for element in reference_mechanism)

        if full:
            selected_mech_ixs = get_full_triangle(reference_indices, side_length)
        else:
            selected_mech_ixs = get_triangle(reference_indices, side_length)
        selected_mechs = nodes_ixs2label(selected_mech_ixs, self.node_labels)
        selected_mechs = ['m_' + s for s in selected_mechs]
        return selected_mechs


    def filter_by_mechanisms(self, selected_mechs, exclude_unselected=False, keep_external_relations=False):
        '''
        Restrict mechanisms in the CES to the selected mechanisms. Either by removing
        the unselected mechanisms (nodes) or my removing the relations (edges) in the
        unselected mechanisms.

        Parameters
        ----------
        selected_mechs : list of mechs (e.g. ['m_GFED', 'm_FEDC'])
        exclude_unselected : bool (whether to exclude nodes and edges or only edges)
        '''
        if 'm' not in selected_mechs[0]:
            selected_mechs = ['m_' + mech for mech in selected_mechs]

        if exclude_unselected:
            self.graph = self.graph.subgraph(selected_mechs)
        else:
            if not keep_external_relations:
                edges_to_remove = [edge for edge in self.graph.edges if (edge[0] not in selected_mechs) or (edge[1] not in selected_mechs)]
                self.graph.remove_edges_from(edges_to_remove)
            else:
                edges_to_remove = [edge for edge in self.graph.edges if
                                   (edge[0] not in selected_mechs) and (edge[1] not in selected_mechs)]
                self.graph.remove_edges_from(edges_to_remove)

    def filter_by_relation_pattern(self, template, pattern_field='pattern'):
        '''
        Returns mech_CES only with relations that match the relation square template.

        Parameters
        ---------
        mech_CES : nx.DiGraph() from CES.get_mech_ces()

        template : Relation square pattern (e.g. '~_~_>_~_~_ ') or list of such patterns (e.g. ['~_~_>_~_~_ ', '~_~_>_ _~_ ']).
        '*' can be used to as wildcard relation (e.g. '~_~_>_*_*_*')

        pattern_field : Key in mech_CES's edge containing relation square pattern.
        '''

        def match_square_rel_patterns(pattern, templates):
            '''
            Checks if relation square match any template in list.
            '''
            matches = [match_square_rel_pattern(pattern, template) for template in templates]
            return np.any(matches)

        def match_square_rel_pattern(pattern, template):
            '''
            Checks if relation square pattern matches template.

            Parameters
            ----------
            pattern : str (e.g. '~_~_>_~_~_ ')
            '''

            if '_' in pattern:
                pattern = pattern.split('_')
            if '_' in template:
                template = template.split('_')

            match = []
            for c1, c2 in zip(pattern, template):
                if c2 == '*':
                    match.append(True)
                elif c1 == c2:
                    match.append(True)
                else:
                    match.append(False)

            return np.all(match)

        if type(template) == str:
            template = [template]

        edge2pattern = nx.get_edge_attributes(self.graph, pattern_field)

        edges_to_remove = [edge for edge, pattern in edge2pattern.items() if
                           not match_square_rel_patterns(pattern, template)]
        self.graph.remove_edges_from(edges_to_remove)

    def print_relation_patterns(self, pattern_field='pattern'):
        '''
        Print relation square patterns in graph edges using 'ces_report' module.
        '''
        
        sorted_patterns, count = self.get_sorted_patterns(pattern_field)
        print(ces_report.underline_str('Relation Patterns'))
        for p, n in zip(sorted_patterns, count):
            print(f"{p} ({n})", sep='\n', end='\n')
            print(ces_report.rel_square_str_body_small(p))
            print()

    def get_sorted_patterns(self, pattern_field='pattern'):
        '''
        Get most frequent relations patterns in graph edges.

        Returns
        -------
        list : sorted patterns
        list : frequency count
        '''

        edge2pattern = nx.get_edge_attributes(self.graph, pattern_field)
        patterns = list(edge2pattern.values())
        pattern_hist = collections.Counter(patterns)
        sorted_patterns, count = zip(*pattern_hist.most_common())
        return sorted_patterns, count

def flip_rel_type(s):
    '''
    Flips relation type (<, >, ~, =).

    Examples
    --------
    >>> flip_rel_type('<')
    '>'
    >>> flip_rel_type('=')
    '='
    '''

    flipper = {'=':'=', '<':'>', '>':'<', '~':'~', ' ':' '}
    return flipper[s]


def flip_rel_square(pattern):
    '''
    Receives a relation square pattern (with order: Ac2eA, Bc2eB, Ac2eB, Ae2cB, Ac2cB, Ae2eB) (e.g. '~_~_~_<_<_<') and returns
    the relation square pattern if the mechanisms related where flipped (A<-->B becomes B<-->A).
    
    Relation squares are flipped in the following way:
    1_2_3_4_5_6 --> 2_1_4'_3'_5'_6' (where apostrophe is the inversion of relation, e.g. < to >)
    
    Examples
    --------
    >>> flip_rel_square('<_=_>_<_=_<')
    '=_<_>_<_=_>'
    
    >>> flip_rel_square('=_<_>_<_=_>')
    '<_=_>_<_=_<'
    
    >>> flip_rel_square('~_=_>_~_>_~')
    '=_~_~_<_<_~'
    '''
    
   
    
    rels = pattern.split('_')
    new_pattern = [rels[1], rels[0], flip_rel_type(rels[3]), flip_rel_type(rels[2]), flip_rel_type(rels[4]), flip_rel_type(rels[5])]
    new_pattern = '_'.join(new_pattern)
    return new_pattern

def color_array(n_colors, n_grays, colormap, exclude_rgb_colors=None):
    '''
    Returns a list of hex colors, where the first n_colors are from the 'colormap' and the last n_grays are tones of gray.

    n_colors : int
    n_grays : int
    colormap : (list of) str
        (list of) colormap name from matplotlib.cm
    exclude_rgb_colors : rgb colors to exclude
    '''
    if type(colormap) != list:
        colormap = [colormap]

    rgb_colors = sum([list(mpl.cm.get_cmap(cm).colors) for cm in colormap], [])
    colors = [mpl.colors.to_hex(color) for color in rgb_colors]

    if exclude_rgb_colors is not None:
        colors = [c for c in colors if c not in exclude_rgb_colors]

    assert n_colors <= len(colors), f"n_colors must be <= {len(colors)} for '{colormap}' colormap."

    grays = get_gray_colors(n_grays)

    return colors[:n_colors] + grays

def get_gray_colors(n_grays, min_black=0, max_white=0.8):
    '''
    Get n gray colors ranging from min_black to max_white.
    '''
    return [mpl.colors.to_hex((n, n, n)) for n in np.linspace(min_black, max_white, n_grays)]

def is_contiguous(x):
    '''
    Checks if indices in array are contiguous (e.g. [3,4,5,6] but not [3,5,6])
    Parameters
    ----------
    x : 1d array of indices

    Returns
    -------
    bool
    '''
    d_x = np.diff(x)
    if np.all(d_x == 1):
        return True
    else:
        return False


def hasse_triangles(n_elements=8):
    '''
    Generate all unit size triangles in the hasse diagram of the contiguous powerset of n_elements.
       /_\
     /_\ /_\
    /_\/_\/_\

    Returns
    -------
    list of 3-tuples

    Example
    -------
    >>> hasse_triangles(3)
    [[(0,), (1,), (0, 1)],
     [(1,), (2,), (1, 2)],
     [(0, 1), (1, 2), (0, 1, 2)],
     [(0, 1), (1, 2), (1,)]]
    '''

    vec = list(range(n_elements))
    vec = [[x] for x in vec]

    powerset = [list(itertools.combinations(range(n_elements), n)) for n in range(1, n_elements + 1)]  # list of lists
    contiguous_sets = [[list(x) for x in sets if is_contiguous(x)] for sets in powerset]

    hotel = {floor: sets for floor, sets in enumerate(contiguous_sets)}  # list of lists

    triangles = []
    for floor in range(0, n_elements - 1):
        rooms = hotel[floor]
        rooms_above = hotel[floor + 1]
        for n in range(n_elements - floor - 1):
            left, right = rooms[n], rooms[n + 1]
            above = rooms_above[n]
            triangles.append(([tuple(left), tuple(right), tuple(above)]))

    for floor in range(1, n_elements):
        rooms = hotel[floor]
        rooms_below = hotel[floor - 1]
        for n in range(n_elements - floor - 1):
            left, right = rooms[n], rooms[n + 1]
            below = rooms_below[n + 1]
            triangles.append([tuple(left), tuple(right), tuple(below)])
    return triangles


def get_triangle(reference_point, side_length, upside_down=False):
    '''
    Returns triangle nodes starting from reference_point (bottom left vertex) in the Hasse diagram.

    Example
    -------
    >>> get_triangle([3,4,5,6], 2, upside_down=False)
    [(3, 4, 5, 6), (5, 6, 7, 8), (3, 4, 5, 6, 7, 8)]

    '''

    assert is_contiguous(reference_point), "Vertex must be of contiguous elements (e.g. [3,4,5])."

    if not upside_down:
        bottom_left = np.array(reference_point)
        bottom_right = bottom_left + side_length
        top = list(range(bottom_left[0], bottom_right[-1] + 1))
        return [tuple(bottom_left), tuple(bottom_right), tuple(top)]
    else:
        bottom = np.array(reference_point)
        top_left = list(range(bottom[-1] - side_length + 1, bottom[-1] + 1))
        top_right = list(range(bottom[0], bottom[0] + side_length))
        return [tuple(bottom), tuple(top_left), tuple(top_right)]


def get_full_triangle(reference_point, side_length):
    '''
    Returns filled-in triangle nodes starting from reference_point (bottom left vertex) in the Hasse diagram.

    Example
    -------
    >>> get_full_triangle((1,2,3), 2)
    [(1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 2, 3, 4), (2, 3, 4, 5), (1, 2, 3, 4, 5)]
    '''
    assert is_contiguous(reference_point), "Inconsistent point given: point must be a contiguous set."

    order = len(reference_point)

    triangle = []
    for n in range(side_length + 1):
        point = move_point_vertically(reference_point, n)
        segment = horizontal_segment_from_point(point, side_length - n)
        triangle += segment
    return triangle


def horizontal_segment_from_point(point, length):
    '''
    Returns horizontal segment from given point in the Hasse diagram
    '''
    assert is_contiguous(point), "Inconsistent point given: point must be a contiguous set."
    segment = [move_point_horizontally(point, n) for n in range(length + 1)]
    return segment


def move_point_horizontally(point, n_steps):
    '''
    Given point in hasse diagram (e.g. (2,3,4)) moves it horizontally.

    Parameter
    ---------
    point : contiguous set
    n_steps : int (positive moves right, negative moves left)

    Example
    -------
    >>> move_point_horizontally((1,2,3), 3)
    (4, 5, 6)
    '''
    assert is_contiguous(point), "Inconsistent point given: point must be a contiguous set."

    return tuple(np.array(point) + n_steps)


def move_point_vertically(point, n_steps, direction='up_right'):
    '''
    Given point in hasse diagram (e.g. (2,3,4)) moves it vertically in the up right direction.

    Parameter
    ---------
    point : contiguous set
    n_steps : int (positive moves right, negative moves left)

    Example
    -------
    >>> move_point_vertically((1,2,3), 3)
    (1, 2, 3, 4, 5, 6)
    '''
    assert is_contiguous(point), "Inconsistent point given: point must be a contiguous set."

    if direction == 'up_right':
        ini = max(point) + 1
        return tuple(list(point) + list(range(ini, ini + n_steps)))
    else:
        raise ValueError('Not implemented.')


def node_ixs2label(ixs, node_labels):
    '''
    Convert node indices to labels.

    Parameters
    ----------
    ixs : tuple of ints
    node_labels : list of str

    Returns
    -------
    str

    Examples
    --------
    >>> node_ixs2label((3,4,5), ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'])
    'EDC'
    '''
    return ''.join(node_labels[ix] for ix in ixs)

def nodes_ixs2label(nodes_ixs, node_labels):
    '''
    Converts list of node indices to labels.

    Parameters
    ----------
    nodes_ixs : list of indices list
    node_labels : list of str

    Returns
    -------
    list of str

    '''
    return [node_ixs2label(ixs, node_labels) for ixs in nodes_ixs]

def node_label2ixs(label, node_labels):
    '''
    Convert node labels to indices.

    Parameters
    ----------
    label : str
    node_labels : list of str

    >>> node_label2ixs('EDC', ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'])
    (3, 4, 5)
    '''
    return tuple([node_labels.index(s) for s in label])



def warp_hasse_layout(node2pos, rho=0.1, mode='exponential'):
    '''
    Warps hasse layout.

    Parameters
    ----------
    node2pos : {node : pos}
    rho : 0 < float

    Returns
    -------
    node2pos
    '''

    def exponential_warp(pos):
        '''
        y = rho * x^2
        '''
        x, y = pos
        xx = x - (end - ini) / 2
        yy = y + rho * xx ** 2
        return x, yy

    def circle_warp(pos):
        '''
        circle: y = - np.sqrt(r^2 - (x - a)^2) + b
        '''
        x, y = pos
        center_x, center_y = center
        radius = center_y / rho
        y_offset = - np.sqrt(radius ** 2 - (x - center_x) ** 2) + center_y
        yy = y + y_offset
        return x, yy

    if rho == 0:
        return node2pos

    pos = np.array([list(pos) for pos in node2pos.values()])
    X, Y = pos[:, 0], pos[:, 1]
    ini, end = min(X), max(X)
    center = ((end - ini) / 2, max(Y))

    if mode == 'exponential':
        return {node: exponential_warp(pos) for node, pos in node2pos.items()}

    elif mode == 'circle':
        return {node: circle_warp(pos) for node, pos in node2pos.items()}
    else:
        raise ValueError("mode must be 'exponential' or 'circle'")

def hasse_layout(n_elements, warp=0, triangle_base=1, warp_mode='exponential'):
    '''
    Generates the Hasse diagram layout out of the contiguous sets in the powerset of 'n_elements'.

    Parameters
    ----------
    n_elements : int
    warp : 0 < float
    triangle_base : float

    Returns
    -------
    dict : {sets : (x,y)}

    '''

    all_sets = [list(itertools.combinations(range(n_elements), n)) for n in range(1, n_elements + 1)]  # list of lists

    contiguous_sets = [[x for x in sets if is_contiguous(x)] for sets in all_sets]

    contiguous_sets_flat = np.sum(contiguous_sets, dtype='object')

    y_scale = triangle_base
    ys = [len(x) * y_scale for x in contiguous_sets_flat]

    xs = []
    x_ini = 0
    for order, sets in enumerate(contiguous_sets):
        xs_tmp = [x_ini + n * triangle_base for n in range(len(sets))]
        xs += xs_tmp
        x_ini += triangle_base / 2

    # xs = list(np.array(xs) / 100)
    # ys = list(np.array(ys) / 100)

    pos = list(zip(xs, ys))

    node2pos = {sets: np.array(xy) for sets, xy in zip(contiguous_sets_flat, pos)}
    if warp != 0:
        node2pos = warp_hasse_layout(node2pos, warp, mode=warp_mode)
    return node2pos


def get_mech_label(mech, kind='mech', add_prefix=True, node_labels=None):
    '''
    Returns the label of the mechanism or of its cause/effect purviews.

    Parameters
    ----------
    mech : pyphi.models.mechanism.Concept
    kind : ['mech', 'cause', 'effect']
    add_prefix : bool
    node_labels : list of str

    Returns
    -------
    str : label

    Examples
    --------
    >>> get_mech_label(mech, 'cause', add_prefix=True)
    'c_FED'
    '''

    if kind.lower() == 'mech':
        ixs = mech.mechanism
        prefix = 'm_'
    elif kind.lower() == 'cause':
        ixs = mech.cause.purview
        prefix = 'c_'
    elif kind.lower() == 'effect':
        ixs = mech.effect.purview
        prefix = 'e_'
    else:
        raise KeyError(f"'{kind}' is not an accepted 'kind'")

    if not add_prefix:
        prefix = ''

    if node_labels is None:
        node_labels = mech.node_labels

    return prefix + ''.join([node_labels[ix] for ix in ixs])


def get_rel_label(rel, kind, node_labels, add_prefix=True):
    '''

    Parameters
    ----------
    rel : pyphi.relations.Relation
    kind : ['mechs', 'purviews', 'rel_purview']
    node_labels : list of str
    add_prefix : bool

    Returns
    -------
    str

    Examples
    --------
    >>> get_rel_labels(rel, 'mechs', node_labels, add_prefix=True)
    ['m_FE', 'm_GFED']

    >>> get_rel_labels(rel, 'purviews', node_labels, add_prefix=True)
    ['c_ED', 'c_FEDC']

    >>> get_rel_labels(rel, 'rel_purview', node_labels, add_prefix=True)
    'p_E'
    '''

    # node_labels = rel.relata[0].node_labels

    def ixs2label(ixs):
        return ''.join([node_labels[ix] for ix in ixs])

    if kind == 'mechs':
        ixs = rel.mechanisms
        prefix = 'm_' if add_prefix else ''
        return [prefix + ixs2label(ixs) for ixs in ixs]

    elif kind == 'purviews':
        if add_prefix:
            type2prefix = {'CAUSE': 'c_', 'EFFECT': 'e_'}
            prefixes = type2prefix[rel.relata[0].direction.name], type2prefix[rel.relata[1].direction.name]
        else:
            prefixes = ['', '']

        ixs = rel.relata.purviews
        return [prefix + ixs2label(ix) for prefix, ix in zip(prefixes, ixs)]

    elif kind == 'rel_purview':
        prefix = 'p_' if add_prefix else ''
        ixs = rel.purview
        return prefix + ixs2label(ixs)

    else:
        raise KeyError(f"'{kind}' is not an accepted 'kind'")


def set_relation(set1, set2):
    """
    Calculates relation of equivalence (=), inclusion (< or >), partial overlap (~) and disjunction ( )between two sets.

    Parameters
    ----------
    set1, set2 : lists

    Returns
    -------
    str : '~', '<', '>', ' ', '='

    Examples
    --------
    >>> set_relation([1,2,3], [2,3,4])
    '~'
    """

    set1, set2 = set(set1), set(set2)
    if set1.intersection(set2) == set([]):
        kind = ' ', 'nothing'
    elif set1 == set2:
        kind = '=', 'equivalence'
    elif set1.issubset(set2):
        kind = '<', 'inclusion'
    elif set2.issubset(set1):
        kind = '>', 'inclusion'
    else:
        kind = '~', 'connection'
    return kind


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v / 256 for v in value]

def plot_colortable(hex_list):
    '''
    Plots a nice color table from hex list.
    '''
    cell_width = 150
    cell_height = 50
    swatch_width = 48
    margin = 12
    topmargin = 40

    rgb_list = [hex_to_rgb(value) for value in hex_list]
    dec_list = [rgb_to_dec(value) for value in rgb_list]
    names = [f'HEX: {col[0]}\nRGB: {col[1]}' for col in zip(hex_list, rgb_list, dec_list)]
    n = len(names)
    ncols = 3
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 8 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=dec_list[i], linewidth=18)
    return fig


def plot_ces(CES, pos=None, fig=None, figsize=(15, 8), edge_color=None, node_color=None,
             node_size=100, with_labels=False, color_key='set_rel_color',
             label_offset=0.07):
    '''
    Plots cause-effect structure using networkx.

    Parameters
    ----------
    CES : CauseEffectStructure()
    pos : dict {node : (x,y)}
    figsize : tuple (width, height)
    node_size : float
    with_labels : bool
    color_key : attribute of edge
    label_offset : float
    '''
    if pos is None:
        pos = nx.get_node_attributes(CES, 'pos')

    if fig is None:
        plt.figure(figsize=figsize)

    if node_color is None:
        node_color = nx.get_node_attributes(CES, 'color').values()
    if edge_color is None:
        edge_color = nx.get_edge_attributes(CES, color_key).values()

    nx.draw(CES, node_size=node_size, node_color=node_color,
            edge_color=edge_color, with_labels=False, pos=pos)

    if with_labels:
        pos_labels = {key: (x, y + label_offset) for key, (x, y) in pos.items()}
        nx.draw_networkx_labels(CES, pos_labels, nx.get_node_attributes(CES, 'name'))

    ax = plt.gca()
    ax.margins(0.20)


def plotly_ces(CES,
               fig=None,
               subplot=(None,None),
               renderer=None,
               edge_color_field='set_rel_color',
               node_color_field='color',
               node_size=10,
               node_edge_width=0.5,
               node_edge_color='black',
               edge_width=1,
               node_opacity=1,
               fontsize=10,
               show_fig=True,
               labels_color='black',
               save_image_path=None,
               save_image_scale=4,
               hide_purview_nodes=False,
               show_purview_label=True,
               show_mech_label=True,
               edge_group_field='rel_type2',
               show_legend=True,
               title='',
               layout_width=1200,
               layout_height=750,
               xlim=None,
               ylim=None):
    '''
    Plots cause-effect structure using Plotly.

    Parameters
    ----------
    CES : CauseEffectStructure()
    subplot : (row, col)
    '''

    if fig is None:
        layout = go.Layout(width=layout_width,
                           height=layout_height,
                           paper_bgcolor='#ffffff',
                           plot_bgcolor='#ffffff')

        fig = go.Figure(layout=layout)

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(range=xlim)
        fig.update_yaxes(range=ylim)

    legendgroups = []

    # NODES


    node_traces = []

    all_nodes = {'purviews': CES.purviews, 'mechs' : CES.mechs}
    show_node_label = {'purviews': show_purview_label, 'mechs' : show_mech_label}

    node_types = ['purviews', 'mechs'] if not hide_purview_nodes else ['mechs']

    for node_type in node_types:
        nodes = all_nodes[node_type]
        mode = 'markers+text' if show_node_label[node_type] else 'markers'

        for node in nodes:
            node_info = CES.nodes[node]

            if show_legend and node_info['type'] not in legendgroups:
                legendgroups.append(node_info['type'])
                show_legend_flag = True
            else:
                show_legend_flag=False

            node_traces.append(go.Scatter(x=[node_info['pos'][0]],
                                        y=[node_info['pos'][1]],
                                        mode=mode,
                                        marker=dict(symbol='circle',
                                                    size=node_size,
                                                    color=node_info[node_color_field],
                                                    line=dict(color=node_edge_color, width=node_edge_width),
                                                    opacity=node_opacity),
                                        hoverinfo='text',
                                        textposition='middle left',
                                        text=node_info['name'],
                                        textfont=dict(color=labels_color,
                                                      size=fontsize),
                                        name=node_info['type'],
                                        legendgroup=node_info['type'],
                                        showlegend=show_legend_flag,
                                        )
                               )

    # EDGES
    node2pos = nx.get_node_attributes(CES, 'pos')

    edge_traces = []
    for kind, edges in zip(['rel', 'mech'], [CES.rel_edges, CES.mech_edges]):
        for edge in edges:
            if kind == 'mech':
                edge_group = 'mechanism'
            else:
                edge_group = CES.edges[edge][edge_group_field]

            # whether to show legend
            if show_legend and edge_group not in legendgroups:
                legendgroups.append(edge_group)
                show_legend_flag = True
            else:
                show_legend_flag = False

            x, y = list(zip(node2pos[edge[0]], node2pos[edge[1]], [None, None]))
            edge_traces.append(go.Scatter(x=x,
                                          y=y,
                                          mode='lines',
                                          line=dict(color=CES.edges[edge][edge_color_field],
                                                    width=edge_width),
                                          hoverinfo='none',
                                          legendgroup=edge_group,
                                          name=edge_group,
                                          showlegend=show_legend_flag)
                               )

    fig.add_traces(edge_traces, rows=subplot[0], cols=subplot[1])
    fig.add_traces(node_traces, rows=subplot[0], cols=subplot[1])
    if title!='':
        fig.update_layout(title_text=title)
    if show_fig:
        fig.show(renderer=renderer)

    if save_image_path is not None:
        fig.write_image(save_image_path, scale=save_image_scale)
    return fig

def plotly_reduced_ces(reduced_CES,
                       renderer=None,
                       show_fig=True,
                       save_image_path=None,
                       save_image_scale=4,
                       layout_width=800,
                       layout_height=600,
                       edge_width=3):
    '''
    Plots reduced cause-effect structure using Plotly
    (no purviews, with collapsed relation square to single edge).

    Parameters
    ----------
    reduced_CES : ReducedCauseEffectStructure()
    '''
    G = reduced_CES.graph

    # LAYOUT
    layout = go.Layout(width=layout_width,
                       height=layout_height,
                       paper_bgcolor='#ffffff',
                       plot_bgcolor='#ffffff')

    fig = go.Figure(layout=layout)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # NODES
    node2pos = nx.get_node_attributes(G, 'pos')
    pos = list(node2pos.values())

    node_names = list(nx.get_node_attributes(G, 'name').values())

    xs, ys = list(zip(*pos))

    node_trace = go.Scatter(x=xs,
                            y=ys,
                            mode='markers+text',
                            marker=dict(symbol='circle',
                                        size=10,
                                        color='#6ea5ff',
                                        line=dict(color='DarkSlateGrey', width=2)),

                            hoverinfo='text',
                            textposition='top center',
                            name='Mechanism',
                            legendrank=0,
                            text=node_names,
                            textfont_size=8
                            )

    # EDGES
    if len(G.edges) > 0:
        all_patterns = nx.get_edge_attributes(G, 'pattern').values()
        top_patterns, _ = zip(*collections.Counter(all_patterns).most_common())

        edge_traces = []
        patterns = []
        for edge in G.edges:
            pattern = G.edges[edge]['pattern']

            if pattern in patterns:
                showlegend = False
                legendrank = 1000
            else:
                showlegend = True
                patterns.append(pattern)
                legendrank = top_patterns.index(pattern) + 1

            x, y = list(zip(node2pos[edge[0]], node2pos[edge[1]], [None, None]))

            edge_trace = go.Scatter(x=x,
                                    y=y,
                                    mode='lines',
                                    line=dict(color=G.edges[edge]['pattern_color'], width=edge_width),
                                    legendgroup=pattern,
                                    showlegend=showlegend,
                                    name=pattern,
                                    legendrank=legendrank,
                                    hoverinfo='none')
            edge_traces.append(edge_trace)

        fig.update_layout(legend_traceorder='grouped')
        fig.add_traces(edge_traces)

    fig.add_trace(node_trace)

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    if show_fig:
        fig.show(renderer=renderer)

    if save_image_path is not None:
        fig.write_image(save_image_path, scale=save_image_scale)
    return fig


def plotly_ces_on_background(CES, CES_background, save_image_path=None):
    '''
    Plots CES with gray pyramid background.

    Parameters
    ----------
    CES : subset of CauseEffectStructure()
    CES_background : full CauseEffectStructure()
    save_image_path : str
    '''

    hasse_ces = CES_background.get_hasse_pyramid()
    nx.set_node_attributes(hasse_ces, LIGHT_GRAY, 'gray_color')
    nx.set_edge_attributes(hasse_ces, 'white', 'gray_color')

    fig = plotly_ces(hasse_ces, edge_color_field='gray_color', node_color_field='gray_color',
                         show_fig=False,
                         labels_color=LIGHT_GRAY,
                         node_opacity=0.5,
                         hide_purview_nodes=True,
                         show_legend=False);

    plotly_ces(CES, fig=fig, edge_width=3, save_image_path=save_image_path);
