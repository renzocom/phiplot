import numpy as np
import pandas as pd
import art

from phiplot import ces_view


# import pyphi.visualize as vis # only for matteo's version

## INDEX ##
# ix2label(node_ixs, node_labels)
# pad_mech_label(label, node_labels)
# set_relation(set1, set2)
# find_ces_pair(mechA, mechB, all_mechs)
# find_rels_pair(mechA, mechB, all_rels)
# calc_rel_square(mechA, mechB, rels, self_rels)
# calc_rule_from_rel_square(rel_square)
# relation_square_str(mechA, mechB, rule)
# calc_rule_from_rel_square(rel_square)
# relation_square_str(mechA, mechB, rule)
# calc_square_rule(mechA, mechB, all_rels)
# calc_ideal_square_rule(mechA, mechB)
# plot_relation_square(labelA, labelB, all_mechs, all_rels, method='real',  print_mechs=False)
# time_relation(mechA, mechB, strict=True)

rel_square_keys = ['Ac2eA', 'Bc2eB', 'Ac2eB', 'Ae2cB', 'Ac2cB', 'Ae2eB']

def quick_distinction_str(distinction):
    ''' Quick and short representation of distinction.


    '''
    mech, cause, effect = ces_view.nodes_ixs2label([distinction.mechanism, distinction.cause_purview, distinction.effect_purview], distinction.node_labels)
    cause_phi, effect_phi = distinction.cause.phi, distinction.effect.phi
    return f"{cause} ({cause_phi:.3f}) -> {mech} -> {effect} ({effect_phi:.3f})"


def ix2label(node_ixs, node_labels):
    """Converts node indices to mechanism label.
    
    Parameters
    ----------
    node_ixs : list/tuple of int
    node_labels : pyphi.labels.NodeLabels or list/tuple of str
    
    Returns
    -------
    str
        
    Example
    -------
    >>> ix2label([0,2], ['A', 'B', 'C'])
    'AC'
    
    >>> type(mech)
    pyphi.models.mechanism.Concept
    
    >>> ix2label(mech.mechanism, mech.node_labels)
    'ABC'
    
    """
    
    return ''.join([node_labels[x] for x in node_ixs])


def frame_str(s):
    'Decorates a string with a #-frame.'
    frame = (4+len(s))*'#'
    s = '# ' + s + ' #'
    new_s = '\n'.join([frame,s,frame])
    return new_s
def underline_str(s, symbol='='):
    return s + '\n' + len(s)*symbol

def pad_mech_label(label, node_labels):
    """
    Pad mechanism label str.
    
    Parameters
    ----------
    node_ixs : str
    node_labels : pyphi.labels.NodeLabels
    
    Returns
    -------
    str
    
    Example
    -------
    >>> pad_mech_label('CD', ['A', 'B', 'C', 'D'])
    '  CD'
    
    >>> pad_mech_label('BD', ['A', 'B', 'C', 'D'])
    ' C D'
    
    """
    s = " " * len(node_labels)
    for c in label:
        ix = node_labels.index(c)
        s = s[:ix] + c + s[ix + 1:]
    return s

def pad_block(s, n, side):
    '''
    Pad block of text with left/right/centered whitespace.
    '''
    if type(s)==str:
        s = s.split('\n')
    if side=='left':
        s = [x.rjust(n) for x in s]
    elif side=='right':
        s = [x.ljust(n) for x in s]
    else: # center
        s = [x.center(n) for x in s]
    
    return '\n'.join(s)

def concat_str_blocks(blocks, interspace):
    '''
    Horizontally concatenate string blocks.
    A block is a string with lines of the same size separated by '\n'.
    '''
    big_block = []
    whitespace = ' '*interspace
    
    for line_blocks in zip(*[block.split('\n') for block in blocks]):
        for i, line_block in enumerate(line_blocks):
            if i==0:
                line = line_block
            else:
                line += whitespace + line_block
        
        big_block.append(line)
    return '\n'.join(big_block)


def remove_common_pad(strs, right=True, left=True):
    '''Removes common padding (left and/or right) for a list of strs.

    Example
    -------
    >>> dist =   '  CD  '
    >>> cause =  ' BC   '
    >>> effect = '   DE '
    >>> remove_common_pad([dist, cause, effect])
    [' CD ', 'BC  ', '  DE']
    '''

    if left:
        while all(s[0] == ' ' for s in strs):
            strs = [s[1:] for s in strs]

    if right:
        while all(s[-1] == ' ' for s in strs):
            strs = [s[:-1] for s in strs]

    return strs


def forget_str(s):
    "Substitutes any non-whitespace for an 'x'."

    if type(s)==list:
        return [''.join([x if x == ' ' else 'x' for x in X]) for X in s]

    return ''.join([x if x == ' ' else 'x' for x in s])

def set_relation(set1, set2):
    """Calculates relation of equivalence (=), inclusion (< or >), partial overlap (~) and disjunction ( )between two sets.
    
    Parameters
    ----------
    set1, set2 : set/list/tuple
    
    Returns
    -------
    str
    
    Examples
    --------
    >>> set_relation([1,2,3], [2,3,4])
    '~'
    """
    
    set1, set2 = set(set1), set(set2)
    if set1.intersection(set2)==set([]):
        kind = ' '
    elif set1 == set2:
        kind = '='
    elif set1.issubset(set2):
        kind = '<'
    elif set2.issubset(set1):
        kind = '>'
    else:
        kind = '~'
    return kind


def get_mech_pair_combs(order, shift, n_nodes=8):
    '''
    Return all combinations of pair of partially overlapping mechanisms in a
    linear network.
    
    Parameters
    ----------
    order : int
        Mechanisms order.
    shift : int
        Overlap shift between mechanisms in the pair.
    n_nodes : int
        Number of nodes in the network.
        
    Returns
    -------
    list : list of pair-list of tuples with mechanism's indices.
    
    Examples
    --------
    >>> get_mech_pair_combs(3, 2, 8)
    [[(0, 1, 2), (2, 3, 4)],
     [(1, 2, 3), (3, 4, 5)],
     [(2, 3, 4), (4, 5, 6)],
     [(3, 4, 5), (5, 6, 7)]]
    '''
    mech_pairs = []
    for n in range(n_nodes - order - shift + 1):
        nodes = list(range(n_nodes))
        x = np.roll(nodes, -n)
        mech_pairs.append([tuple(x[:order]), tuple(x[shift:shift+order])])
    return mech_pairs

def mech_str(mech, horizontal, remove_pad=False, forget=False):
    '''
    Returns str with mechanism representation.
    
    Parameters
    ----------
    mech : pyphi.models.mechanism.Concept
    horizontal : bool
        Whether mechanism is formatted in vertical or horizontal orientation
    remove_pad : bool
    forget : bool
        Changes letters for x's
    '''
    
    dist = pad_mech_label(ix2label(mech.mechanism, mech.node_labels), mech.node_labels)
    cause = pad_mech_label(ix2label(mech.cause_purview, mech.node_labels), mech.node_labels)
    effect = pad_mech_label(ix2label(mech.effect_purview, mech.node_labels), mech.node_labels)
       
    if forget:
        dist, cause, effect = forget_str([dist, cause, effect])
    
    if horizontal:
        if remove_pad:
            dist, cause, effect = remove_common_pad([dist, cause, effect])
        return "{:} : {:} <- {:}".format(dist, effect, cause)
    else:
        dist, cause, effect = remove_common_pad([dist, cause, effect])
        n = np.max([len(dist), len(cause), len(effect)])
        dash = '-' * n
        return "{:}\n{}\n{:}\n{:}".format(dist, dash, effect, cause)
    
def multi_mech_str(mechs, horizontal, remove_pad):
    dists = [pad_mech_label(ix2label(mech.mechanism, mech.node_labels), mech.node_labels) for mech in mechs]
    causes = [pad_mech_label(ix2label(mech.cause_purview, mech.node_labels), mech.node_labels) for mech in mechs]
    effects = [pad_mech_label(ix2label(mech.effect_purview, mech.node_labels), mech.node_labels) for mech in mechs]
    
    if horizontal:
        S = []
        for n, (dist, cause, effect) in enumerate(zip(dists, causes, effects)):
            
            if remove_pad:
                dist, cause, effect = remove_common_pad([dist, cause, effect])
                
            s = "{:} : {:} <- {:}".format(dist, effect, cause)
            S.append(s)
        
        return ' | '.join(S)
    
    else: # vertical
        for n, (dist, cause, effect) in enumerate(zip(dists, causes, effects)):
            dists[n], causes[n], effects[n] = remove_common_pad([dist, cause, effect])

        N = [np.max([len(dist), len(cause), len(effect)]) for dist, cause, effect in zip(dists, causes, effects)]
        dash = ['-' * n for n in N]
        D, E, C = ' | '.join(dists), ' | '.join(effects), ' | '.join(causes)
        dashes = '-|-'.join(dash)
        return "{}\n{}\n{}\n{}\n".format(D, dashes, E, C)
    
def rel_square_str(mechA, mechB, rule):
    '''
    Returns the full str representation of the square of set-relations between
    two mechanisms.
    
    Parameters
    ----------
    mechA, mechB : pyphi.models.mechanism.Concept
    rule : str
        String with relation square between two mechanisms.
        
    Examples
    --------
    >>> print(rel_square_str(mechA, mechB, ' _=_~_ _ _ '))
    
    EDC           CBA
    ---  ┌ - - ┐  ---
    EDC   \   /|  CBA
         |   ~ |     
            x  =     
         | /   |     
          /   \|     
    ED   └ - - ┘  CBA
    
    '''
    mechs = [mechA, mechB]
    dists = [pad_mech_label(ix2label(mech.mechanism, mech.node_labels), mech.node_labels) for mech in mechs]
    causes = [pad_mech_label(ix2label(mech.cause_purview, mech.node_labels), mech.node_labels) for mech in mechs]
    effects = [pad_mech_label(ix2label(mech.effect_purview, mech.node_labels), mech.node_labels) for mech in mechs]
    
    inter = 2
    
    body = rel_square_str_body_small(rule).split('\n') # 7 lines
    
    N = [[],[]]
    cols = [[],[]]
    for n in range(2):
        dists[n], causes[n], effects[n] = remove_common_pad([dists[n], causes[n], effects[n]])
        N[n] = np.max([len(dists[n]), len(causes[n]), len(effects[n])])
        
        cols[n] = ['-'*N[n], effects[n]] + [' '*N[n]]* 4 + [causes[n]]
    
    cols[0] = [x + ' '*inter for x in cols[0]]
    cols[1] = [' '*inter + x for x in cols[1]]
        
    header = '\n'.join([dists[0] + ' '*(2*inter + 7) + dists[1]])
    
    text = []
    for m in range(7):
        text.append(cols[0][m] + body[m] + cols[1][m])
        
    text = '\n'.join(text)
    
    return '\n'.join([header, text])

def rel_square_str_big(mechA, mechB, rule):
    """
    [OLD]
    Returns the full str representation of the square of set-relations between
    two mechanisms.
    """
    
    mechA_label = ''.join([mechA.node_labels[x] for x in mechA.mechanism])
    causeA_label = ''.join([mechA.node_labels[x] for x in mechA.cause_purview])
    effectA_label = ''.join([mechA.node_labels[x] for x in mechA.effect_purview])

    mechB_label = ''.join([mechB.node_labels[x] for x in mechB.mechanism])
    causeB_label = ''.join([mechB.node_labels[x] for x in mechB.cause_purview])
    effectB_label = ''.join([mechB.node_labels[x] for x in mechB.effect_purview])

    # A <--> B
    #     A c-e A, B c-e B, A c-e B, B c-e A, A c-c B, A e-e B
    # rule = ['>', '>', '>', ' ', '~', ' ']
    
    print("Rule: ", "_".join(rule))
    print()
    
    header = \
    "{:5}   {:>5}   distinctions\n".format(mechA_label, mechB_label) + \
    "_____________\n\n" + \
    "{:5}   {:>5}   effect purviews\n".format(effectA_label, effectB_label)

    body = rel_square_str_body_big(rule)
        

    tail = "{:5}   {:>5}   cause purviews\n".format(causeA_label, causeB_label)
    
    return header + body + tail

def pretty_rel_square_str_body_small(rule):
    s = rel_square_str_body_small(rule).split('\n')
    head = 'mech   A     B'
    s[0] =  'effect   '+ s[0]
    s[6] =  'cause   '+ s[6]
    ss = '\n'.join([head] + s)
    return pad_block(ss, 18, side='left')
    
def rel_square_str_body_small(rule):
    """Returns the str representation of a square rule (small version).
    
    Parameters
    ----------
    rule : str or list
        List containing relation rule between two mechanisms.
        (order: Ac2eA, Bc2eB, Ac2eB, Ae2cB, Ac2cB, Ae2eB)
        
    Example
    -------
    >>> print(rel_square_str_body_big('=_=_<_<_<_<'))
    ┌--<--┐
    |\   /|
    | \ < |
    =  x  =
    | / > |
    |/   \|
    └--<--┘
    """
    
    if len(rule)>6:
        rule = rule.split('_')
    
    if rule[5]==' ':
        body = "┌ - - ┐\n".format("~")
    elif rule[5]=='':
        body = "┌     ┐\n".format("~")
    else:
        body = "┌--{}--┐\n".format(rule[5])

    left  = ' ' if (rule[0]==' ') or (rule[0]=='') else '|'
    right = ' ' if (rule[1]==' ') or (rule[1]=='') else '|'

    left_diag = ' ' if (rule[3]==' ') or (rule[3]=='') else '\\'
    right_diag = ' ' if (rule[2]==' ') or (rule[2]=='') else '/'

    left2 = ' ' if (rule[0]=='') else '|'
    right2 = ' ' if (rule[1]=='') else '|'
    left_diag2 = ' ' if (rule[3]=='') else '\\'
    right_diag2 = ' ' if (rule[2]=='') else '/'


    horizontal2vertical = {'<' : 'V', '>' : 'A', '~' : 's', '=':'=', ' ':' ', '':' '}
    left_sym = horizontal2vertical[rule[0]]
    right_sym = horizontal2vertical[rule[1]]

    left_diag_sym = rule[3]
    right_diag_sym = rule[2]
    # if rule[3] == '<':
    #     left_diag_sym = '>'
    # elif rule[3] == '>':
    #     left_diag_sym = '<'
    
    body += \
    "{}{}   {}{}\n".format(left, left_diag2, right_diag2, right) + \
    "{} {} {} {}\n".format(left2, left_diag, right_diag_sym, right2) + \
    "{}  x  {}\n".format(left_sym, right_sym) + \
    "{} {} {} {}\n".format(left2, right_diag, left_diag_sym, right2) + \
    "{}{}   {}{}\n".format(left, right_diag2, left_diag2, right)
    if rule[4]==' ':
        body += "└ - - ┘".format("~")
    elif rule[4]=='':
        body += "└     ┘".format("~")
    else:
        body += "└--{}--┘".format(rule[4])
        
    return body

def rel_square_str_body_big(rule):
    """Returns the str representation of a square rule (big version).
    
    Parameters
    ----------
    rule : str or list
        List containing relation rule between two mechanisms
        (order: Ac2eA, Bc2eB, Ac2eB, Ae2cB, Ac2cB, Ae2eB)
    
    Example
    -------
    >>> print(rel_square_str_body_big('=_=_<_<_<_<'))
     --- < ----
    |\        /|
    | \      / |
    |  >    <  |
    |   \  /   |
    =    \/    =
    |    /\    |
    |   /  \   |
    |  /    \  |
    | /      \ |
    |/        \|
     --- < ----
    """
    if len(rule)>6:
        rule = rule.split('_')
    
    if rule[5]==' ':
        body = " - - - - -\n".format("~")
    else:
        body = " --- {} ----\n".format(rule[5])

    l  = ' ' if rule[0]==' ' else '|'
    r = ' ' if rule[1]==' ' else '|'
    left_diag = ' ' if rule[3]==' ' else '\\'
    right_diag = ' ' if rule[2]==' ' else '/'

    horizontal2vertical = {'<' : 'V', '>' : 'A', '~' : 's', '=':'=', ' ':'|'}
    left_sym = horizontal2vertical[rule[0]]
    right_sym = horizontal2vertical[rule[1]]

    left_diag_sym = rule[3]
    right_diag_sym = rule[2] # invert orientation
    if rule[3]=='<':
        left_diag_sym = '>'
    elif rule[3]=='>':
        left_diag_sym = '<'
    
    body += \
    "|{}        {}|\n".format(left_diag, right_diag) + \
    "{} \      / {}\n".format(l, r) + \
    "|  {}    {}  |\n".format(left_diag_sym, right_diag_sym) + \
    "{}   \  /   {}\n".format(l, r) + \
    "{}    {}{}    {}\n".format(left_sym, left_diag, right_diag, right_sym) + \
    "{}    {}{}    {}\n".format(l, right_diag, left_diag, r) + \
    "|   /  \   |\n" + \
    "{}  {}    {}  {}\n".format(l, right_diag, left_diag, r) + \
    "| /      \ |\n" + \
    "{}{}        {}{}\n".format(l, right_diag, left_diag, r)

    if rule[4]==' ':
        body += " - - - - -\n".format("~")
    else:
        body += " --- {} ----\n".format(rule[4])
        
    return body

def calc_rel_square_rule(mechA, mechB, rel_square=None, exact=True):
    """ Computes the kind of set relation of a pair of mechanisms. The self-relations
    (Ac2eA and Bc2eB) and cross-relations (Ac2cB, Ac2eB, etc) are classified according
    to inclusion, partial overlap, equivalence and none.
    
    Parameters
    ----------
    mechA, mechB : pyphi.models.mechanism.Concept
    rel_square : dict with Ac2eA, Bc2eB, Ac2eB, Ae2cB, Ac2cB, Ae2eB as keys and relations as values (see get_rel_square).
    ideal : bool
        Ignores the phi-relation, i.e. evaluates set-relations looking just at the purview.
    
    Returns
    -------
    dict : dictionary with Ac2eA, Bc2eB, Ac2eB, Ae2cB, Ac2cB, Ae2eB as keys and set relations ('>', '<', '=', '~' or ' ') as values 
    
    """
    if exact:
        rule = {}
        for kind, (rel, right_direction) in rel_square.items():
            if rel is None:
                rule[kind] = ' '
            else:
                if right_direction:
                    rule[kind] = set_relation(rel.relata[0].purview, rel.relata[1].purview)
                else:
                    rule[kind] = set_relation(rel.relata[1].purview, rel.relata[0].purview)
    
    else: # approximate/ideal
        rel_square_ideal = {'Ac2eA': [mechA.cause_purview, mechA.effect_purview],
                     'Bc2eB': [mechB.cause_purview, mechB.effect_purview],
                     'Ac2eB': [mechA.cause_purview, mechB.effect_purview],
                     'Ae2cB': [mechA.effect_purview, mechB.cause_purview],
                     'Ac2cB': [mechA.cause_purview, mechB.cause_purview],
                     'Ae2eB': [mechA.effect_purview, mechB.effect_purview]}
        rule = {}
        for kind, purviews in rel_square_ideal.items():
            set_rel = set_relation(purviews[0], purviews[1])
    #         print(purviews, kind)
            rule[kind] = set_rel
    
    return rule

def get_rel_square(mechA, mechB, all_rels):
    '''Given pair of mechanisms and list of relations finds self-relations and relations
    between.
    
    Parameters
    ----------
    mechA, mechB : mechanism indices
    
    rels : list
        List of pyphi.relations.Relation

    Returns
    -------
    dict 
        Dict with Ac2eA, Bc2eB, Ac2eB, Ae2cB, Ac2cB, Ae2eB fields.
    
    '''
    def is_right_direction(mechA, mechB, rel):
        '''
        Checks if relation is between mechA and mechB (returns True), or between
        mechB and mechA (returns False).
        '''

        assert set(rel.mechanisms) == set([mechA.mechanism, mechB.mechanism]), 'Relation does not match mechanism A and B'

        if mechA.mechanism == rel.relata[0].mechanism and mechB.mechanism == rel.relata[1].mechanism:
            return True
        if mechA.mechanism == rel.relata[1].mechanism and mechB.mechanism == rel.relata[0].mechanism:
            return False
        print("Ops, this shouldn't happen!")
        return 2

    rel_square = {'Ac2eA':[], 'Bc2eB':[], 'Ac2eB':[], 'Ae2cB':[], 'Ac2cB':[], 'Ae2eB':[]}
    
    for rel in all_rels: # appends a tuple with (relation, bool (right direction))
        if set([mechA.mechanism]) == set(rel.mechanisms): # self-relation
            rel_square['Ac2eA'].append((rel, True))
        elif set([mechB.mechanism]) == set(rel.mechanisms): # self-relation
            rel_square['Bc2eB'].append((rel, True))
        elif set([mechA.mechanism, mechB.mechanism]) == set(rel.mechanisms): # cross-relation
            if is_right_direction(mechA, mechB, rel):
                ix1, ix2 = 0, 1
                flag = True
            else:
                ix1, ix2 = 1, 0
                flag = False
    
            if str(rel.relata[ix1].direction) == 'CAUSE' and str(rel.relata[ix2].direction) == 'EFFECT':
                rel_square['Ac2eB'].append((rel, flag))
            elif str(rel.relata[ix1].direction) == 'EFFECT' and str(rel.relata[ix2].direction) == 'CAUSE':
                rel_square['Ae2cB'].append((rel, flag))
            elif str(rel.relata[ix1].direction) == 'CAUSE' and str(rel.relata[ix2].direction) == 'CAUSE':
                rel_square['Ac2cB'].append((rel, flag))

            elif str(rel.relata[ix1].direction) == 'EFFECT' and str(rel.relata[ix2].direction) == 'EFFECT':
                rel_square['Ae2eB'].append((rel, flag))
    
    for key, val in rel_square.items():
        assert len(val)==0 or len(val)==1, "Relations given contain duplicates in {key}: {val}"
        if len(val)==1:
            rel_square[key] = val[0]
        else: # empty
            rel_square[key] = (None, None)
            
    return rel_square

def time_relation(mechA, mechB):
    """
    Assuming the following time network:
     H ← G ← F ← E ← D ← C ← B ← A
     \_______________________/   |
            RETENTIONS          NOW
            
                     ----->
                     (flow)

    Returns kind of temporal relation T(A, B) (succession, termination or succession)
    between distinctions A and B.
    
    Succession: "B succeeds A" if there is partial overlap between the distinctions and
    NOW(B) > NOW(A), e.g. A='DCB' and B='CBA'
    Presentification: "A presentifies B" if A is included in B and includes the NOW(B), e.g. CB and EDCB
    Retention: "A retains B" if A is included in B and doesn't include NOW(B), e.g. EDC and EDCB
        
    Parameters
    ----------
    mechA, mechB : pyphi.models.mechanism.Concept
    struct : bool
    
    Returns
    -------
    str : 'successsion', 'presentification', 'retention' with prefix 'flip' is relation order is reversed
    dict : dictionary with statistics
    """
    A, B = tuple(mechA), tuple(mechB)
    setA, setB = set(A), set(B)
    
    order_diff = abs(len(A) - len(B))
    min_shift = min(len(setA.difference(setB)), len(setB.difference(setA)))
    max_shift = max(len(setA.difference(setB)), len(setB.difference(setA)))
    overlap = len(setA.intersection(setB))
    
    stats = {'order_diff' : order_diff, 'overlap' : overlap,
            'min_shift' : min_shift, 'max_shift' : max_shift, }
    
    if setA.intersection(setB) == set([]):
        return 'none', stats
    
    if setA==setB:
        return 'identity', stats
    
    flag = ''
    if setB.issubset(setA):
        flag = 'flip_'
        setA, setB, A, B = setB, setA, B, A
    
    if setA.issubset(setB):
        if B[-1] in setA: # A includes NOW(B)
            return flag + 'presentification', stats
        
        else:
            return flag + 'retention', stats
        
    else: # partial overlap
        if max(A) > max(B):
            flag = 'flip_'
            setA, setB, A, B = setB, setA, B, A
            
        return flag + 'succession', stats
       

def create_ces_table(ces, rels):

    from itertools import combinations

    # get all pairs of mechanisms
    print('Getting pairs of mechanisms')
    all_mech_pairs = list(combinations(ces, 2))

    # fix orientation of time relation by flipping A and B
    # default: given A and B, either A precedes B, or A  
    print('Fixing orientation')
    for n, (mechA, mechB) in enumerate(all_mech_pairs):
        time_rel = time_relation(mechA.mechanism, mechB.mechanism)[0]
        if time_rel.split('_')[0] == 'flip':
            all_mech_pairs[n] = (all_mech_pairs[n][1], all_mech_pairs[n][0])

    all_mechA = [a for a,b in all_mech_pairs]
    all_mechB = [b for a,b in all_mech_pairs]

    df = pd.DataFrame(index=range(len(all_mech_pairs)))

    ##############
    # MECHANISMS #
    ##############

    # labels
    print('Setting mech labels')
    df['A'] = [ix2label(a.mechanism, a.node_labels) for a,b in all_mech_pairs]
    df['B'] = [ix2label(b.mechanism, b.node_labels) for a,b in all_mech_pairs]
    df['causeA'] = [ix2label(a.cause_purview, a.node_labels) for a,b in all_mech_pairs]
    df['effectA'] = [ix2label(a.effect_purview, a.node_labels) for a,b in all_mech_pairs]
    df['causeB'] = [ix2label(b.cause_purview, b.node_labels) for a,b in all_mech_pairs]
    df['effectB'] = [ix2label(b.effect_purview, b.node_labels) for a,b in all_mech_pairs]

    # indices
    print('Setting mech indices')
    df['ixsA'] = [a.mechanism for a,b in all_mech_pairs]
    df['ixsB'] = [b.mechanism for a,b in all_mech_pairs]
    df['cause_ixsA'] = [a.cause_purview for a,b in all_mech_pairs]
    df['effect_ixsA'] = [a.effect_purview for a,b in all_mech_pairs]
    df['cause_ixsB'] = [b.cause_purview for a,b in all_mech_pairs]
    df['effect_ixsB'] = [b.effect_purview for a,b in all_mech_pairs]

    ######################
    # TEMPORAL RELATIONS #
    ######################
    print('Setting set relations...')
    # INCLUSION AND PARTIAL OVERLAP (SET RELATIONS)
    sym2name = {'<' : 'inclusion', '>' : 'inclusion', '~' : 'overlap', '=' : 'identity', ' ' : 'none'}
    df['set_rel'] = [sym2name[set_relation(mechA.mechanism, mechB.mechanism)] for mechA, mechB in all_mech_pairs]

    # SUCCESSION, RETENTION, PRESENTIFICATION
    print('Setting time relations...')
    X = [time_relation(mechA.mechanism, mechB.mechanism) for mechA, mechB in all_mech_pairs]
    time_rels, stats = [x[0] for x in X], [x[1] for x in X]
    df['time_rel'] = time_rels
    for stat in stats[0].keys():
        df[stat] = [x[stat] for x in stats]

    # STRICT SUCCESSION, STRICT RETENTION, PRESENTIFICATION (SAME ORDER SUCCESSION, LAST MOMENT RETENTION)
    print('Setting time relations (strict)...')
    df['strict_time_rel'] = 'none'
    succession_mask = (df['time_rel']=='succession') & (df['order_diff']==0)
    df.loc[succession_mask, 'strict_time_rel'] = 'succession'
    df.loc[df['time_rel']=='presentification', 'strict_time_rel'] = 'presentification'
    last_moment_condition = np.array([b[0] in a for a,b in zip(df.ixsA, df.ixsB)])
    retention_mask = (df['time_rel']=='retention') & last_moment_condition
    df.loc[retention_mask, 'strict_time_rel'] = 'retention'

    # VERY STRICT SUCCESSION, STRICT RETENTION, PRESENTIFICATION (1-SHIFT)
    print('Setting time relations (very strict)...')
    df['verystrict_time_rel'] = 'none'
    succession_mask = (df['time_rel']=='succession') & (df['order_diff']==0) & (df['max_shift']==1)
    df.loc[succession_mask, 'verystrict_time_rel'] = 'succession'
    presentification_mask = (df['time_rel']=='presentification') & (df['max_shift']==1)
    df.loc[presentification_mask, 'verystrict_time_rel'] = 'presentification'
    retention_mask = (df['time_rel']=='retention') & np.array([b[0] in a for a,b in zip(df.ixsA, df.ixsB)]) & (df['max_shift']==1)
    df.loc[retention_mask, 'verystrict_time_rel'] = 'retention'

    # Get relation squares
    n_pairs = len(all_mech_pairs)
    print(f'Calculating {n_pairs} rules...')

    rel_squares = []
    rules = []
    for n, (mechA, mechB) in enumerate(all_mech_pairs):
        if n % int(n_pairs / 100) == 0:
            print('#', end='')
        
        rel_square = get_rel_square(mechA, mechB, rels) # dictionary
        rule = calc_rel_square_rule(mechA, mechB, rel_square, exact=True) # dictionary
        rules.append(rule)
        rel_squares.append(rel_square)

    
    print()

    # print('Calculating approx rules...')
    # approx_rules = ["_".join(calc_rel_square_rule(mechA, mechB, exact=False)) for mechA, mechB in all_mech_pairs]
    # df['approx_rule'] = approx_rules

    # print('Calculatin match...')
    # match = []
    # for rule, ideal in zip(rules, approx_rules):
    #     match.append(sum([1 if r==i else 0 for r,i in zip(rule, ideal)]))
    # df['match'] = match

    df['mechA'] = all_mechA
    df['mechB'] = all_mechB

    # add square relations
    for label in rel_square_keys:
        df[label] = [rel_square[label] for rel_square in rel_squares]

    # add set square relations
    for label in rel_square_keys:
        df[f'set_{label}'] = [rule[label] for rule in rules]
    
    # add set square rule
    df['rule'] = ["_".join(rule.values()) for rule in rules]

    print('Done!')
    return df

def section_str(s):
    divider = divider_str()
    return '\n' + divider + '\n' + s + '\n' + divider + '\n'

def subsection_str(s):
    divider = divider_str()
    return '\n' + s + '\n' + divider + '\n'

def divider_str(s='-'):
    return s*100

def bigtitle_str(s):
    return art.text2art('Time Network')

def write_runinfo(run_id, params):
    '''
    Writes header with basic information of the CES analysis.
    '''
    width = 80
    run_str = f"run id: {run_id}".center(width)

    par_str = ''
    for key,val in params.items():
        par_str += f"{key} = {val}  "
    par_str = par_str.center(width)

    network_diagram = "H ← G ← F ← E ← D ← C ← B ← A"

    text = \
    f'''
{run_str}

{par_str}

{network_diagram.center(width)}
    '''

    return text

def write_mech_patterns(mechs):
    '''
    Writes most frequent overlap patterns in the CES between distinction & cause/effect purviews.
    '''
    # create dataframe with mech-effect-cause overlap patterns
    patterns = []
    orders = []
    for mech in mechs:
        orders.append(len(mech.mechanism))
        patterns.append(mech_str(mech, False, forget=True))
    pattern_df = pd.DataFrame({'pattern':patterns, 'order':orders, 'mech':mechs, })

    # quantify relative frequency of each pattern by mechanism order
    order_count = pattern_df['order'].value_counts()
    pattern_count = {}
    for order, dff in pattern_df.groupby('order'):
        tmp = dff['pattern'].value_counts() / order_count[order]
        pattern_count[order] = tmp.sort_values(ascending=False)

    # recover example of most frequent pattern
    order2pattern_freq = {}
    for order in pattern_count.keys():
        x_pattern = pattern_count[order].index[0]
        mech = pattern_df[pattern_df.pattern==x_pattern].mech.values[-1]
        pattern = mech_str(mech, horizontal=False, forget=False)

        freq = pattern_count[order][0]
        order2pattern_freq[order] = (pattern, freq)

    # generate summary
    blocks = []
    for order, (pattern, freq) in order2pattern_freq.items():
        
        width = max(3, len(pattern.split('\n')[0]))
        l1 = f"{order}".center(width)
        l2 = f"{freq*100:.0f}%".center(width)
        l3 = pad_block(pattern, 3, 'right')

        s = "\n".join([l1, '', l2,'', l3])
        blocks.append(s)

    infoblock = pad_block('\n'.join(['order', '' , 'freq', '', 'mech', '', 'effect', 'cause']), 8, 'left')
    blocks = [infoblock] + blocks

    text = '\n' + concat_str_blocks(blocks, 5) + '\n'
    return text

def write_mechanisms(mechs, by_order):
    '''
    Prints list of mechanisms in the CES.
    '''
    text = ''
    if not by_order:
        mech_header = f"{'mech'.center(8)} : {'effect'.center(8)} <- {'cause'.center(8)}"
        text += f"\n\n{underline_str(mech_header, '-')}"
        for mech in mechs:
            dist = pad_mech_label(ix2label(mech.mechanism, mech.node_labels), mech.node_labels)
            cause = pad_mech_label(ix2label(mech.cause_purview, mech.node_labels), mech.node_labels)
            effect = pad_mech_label(ix2label(mech.effect_purview, mech.node_labels), mech.node_labels)
            text += "\n{:8} : {:8} <- {:8}".format(dist, effect, cause)
    else:   
        order = 1
        for i, mech in enumerate(mechs):
            if len(mech.mechanism)>=order:
                mech_vert = pad_block(mech_str(mech, horizontal=False, remove_pad=False), 28, 'center')
                mech_header = f"{'mech'.center(8)} : {'effect'.center(8)} <- {'cause'.center(8)}"
                mech_header = underline_str(mech_header, '-')

                text += \
f'''
{underline_str(f"order: {order}", '=')}
{mech_vert}

{mech_header}\n'''
                order+=1
                
            text += mech_str(mech, horizontal=True, remove_pad=False) + '\n'
        text += '\n'
    return text
            
def get_typical_rules(df, time_rel_col, time_rel_types, prefix=''):
    '''
    Calculates the typical rule.
    '''
    typical_rules = {}
    for time_rel_type in time_rel_types:
        df_type = df[df[time_rel_col]==time_rel_type]

        typical_rule = []
        for n, key in enumerate(rel_square_keys):
            set_rel = df_type[prefix + key].value_counts().index[0]
            typical_rule.append(set_rel)

        typical_rules[time_rel_type] = typical_rule
    return typical_rules

def write_rel_squares(df, time_rel_col, time_rel_types, summary=True):
    # statistics on time relation scheme

    counts = df[time_rel_col].value_counts()
    n_square_rels = counts.sum()
    n_time_rels = counts[time_rel_types].sum()
    percentage = 100*(n_time_rels/n_square_rels)

    s = f'\nTime Relation Scheme: {time_rel_col.upper()}'
    s += f'({n_time_rels} out of all {counts.sum()} relation pairs ({percentage:.0f}%))'
    text = underline_str(s, '=') + '\n\n'
    
    # s = ''
    # for key in time_rel_types:
        
    #     s += f"{key:<22} {counts[key]:>4}\n"

    # s += f"{'':<22} ----\n"
    # s += f"{'temporal':<22} {n_time_rels:>4} out of {counts.sum():>4} relation squares ({percentage:.0f}%)\n\n\n"
    # text += s

    if summary:
        title = [underline_str((rel_type.upper() + f' ({counts[rel_type]}/{n_time_rels})').center(22), '.') for rel_type in time_rel_types]
        text += concat_str_blocks(title, 12) + '\n'
        for n in range(2):
            blocks = []
            count = []
            for rel_type in time_rel_types:
                df_type = df[df[time_rel_col]==rel_type]
                if len(df_type)>0:
                    value_counts = df_type['rule'].value_counts()
                    rules = value_counts.index.values
                    vals = value_counts.values
                    total = vals.sum()
                    count.append(100*(vals[n]/total))

                    rule = rules[n]
                    df_rule = df_type[df_type['rule']==rule]
                    mechA, mechB = df_rule.mechA.iloc[n], df_rule.mechB.iloc[n]
                    block = pad_block(rel_square_str(mechA, mechB, rule), 22, side='center')
                    blocks.append(block)

            lens = [len(block.split('\n')[0]) for block in blocks]
            subtitle = [f'{c:.1f}%'.center(le) for c, le in zip(count, lens)]
            text += concat_str_blocks(subtitle, 12) + '\n'
            text += concat_str_blocks(blocks, 12) + '\n'

    else:
        for rel_type in time_rel_types:
            text += underline_str(rel_type.upper(), '.') + '\n\n'
            df_type = df[df[time_rel_col]==rel_type]
            for n, (rule, count) in enumerate(df_type['rule'].value_counts()[:3].items()):
                prettyrule = ', '.join(rule.split('_'))
                text += f"#{n+1} rule: [{prettyrule}] | count: {count}\n"
                df_rule = df_type[df_type['rule']==rule]
                mechsA, mechsB = df_rule['mechA'].values, df_rule['mechB'].values
                for i, (mechA, mechB) in enumerate(zip(mechsA, mechsB)):
                    if i==0:
                        text += \
f"""
{pad_block(rel_square_str(mechA, mechB, rule), 50, side='center')}

dist   :  effect  <-  cause   |   dist   :  effect  <-  cause   
{'-'*66}
"""
                    text += multi_mech_str([mechA, mechB], horizontal=True, remove_pad=False) + '\n'
                text += '\n'
    return text
                
def write_rel_squares_byrel(df, time_rel_col, time_rel_types):
    '''
    Writes relation square statistics as average of individual relation distribution.
    '''
    
    prefix = ''

    text = ''
    typical_rules = get_typical_rules(df, time_rel_col, time_rel_types)
    for rel_type in time_rel_types:

        df_type = df[df[time_rel_col]==rel_type]
        
        typical_rule = typical_rules[rel_type]

        text += \
f'''\n{underline_str(rel_type.upper() + f' ({len(df_type)}/{len(df)})', '.')} 

{pretty_rel_square_str_body_small(typical_rule)}
              ↳ average relation square

Distribution of set-relations across square:\n'''

        for n, key in enumerate(rel_square_keys):
            hist = df_type[prefix + key].value_counts()
            symbols = hist.index.values
            count = hist.values
            total = np.sum(count)

            s = ''
            for symbol, n in zip(symbols, count):
                percent = 100 * (n / total)
                s += (f"'{symbol}' : {percent:.0f}% , ")
            text += f"{key} : {s}\n"
        text += '\n'
    return text