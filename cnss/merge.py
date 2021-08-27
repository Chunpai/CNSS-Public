import collections
import copy

import numpy as np
import networkx as nx
from multiprocessing import Pool, Lock
import logging

output_lock = Lock()


def find_tree_subgraph(G, core_dict, tree_visited, root, alpha):
    """
    find tree subgraph that the root is in the core,
    use BFS to find all adjacent positive periphery tree nodes

    :param G:
    :param core_dict: a dictionary of core nodes
    :param tree_visited: used to track the tree nodes which have been explored
    :param root: here denotes a node in the core
    :param alpha: significant threshold

    :return T_subgraph: a explored tree subgraph (networkx graph object which),
    :return T_positive: all positive nodes in the explored tree, including the root in the core

    """
    T_visited = {root: 1}
    queue = [root]
    T_positive = {}
    if G.nodes[root]['p'] <= alpha:
        T_positive[root] = 1
    while len(queue) != 0:
        neighbors_list = list(G.neighbors(queue.pop(0)))
        for nei in neighbors_list:
            if nei not in core_dict and nei not in tree_visited and G.nodes[nei]['p'] <= alpha:
                queue.append(nei)
                tree_visited[nei] = 1
                T_visited[nei] = 1
                T_positive[nei] = 1
    T_nodes = T_visited.keys()
    # we need copy the subgraph, since the subgraph passed by reference
    T_subgraph = G.subgraph(T_nodes).copy()
    return T_subgraph, tree_visited, T_positive


def compress(G, C, alpha):
    """
    compress the periphery tree nodes, and return the core as
    a networkx Graph data object with compressed tree-nodes

    @param G: networkx graph object with nodes attributes contained
    @param C:
    :return: a compressed core
    """
    p_values_dict = dict(G.nodes(data="p", default=1))
    core_dict = dict([(n, 1) for n in C.nodes()])

    pos_count = 0
    neg_count = 0
    G2 = G.copy()  # the G will be used later, and G2 will be changed.
    tree_visited = {}
    roots = []

    # remove isolated core nodes
    # if a core node has no neighbor in the core, then remove that node from the core dict
    # since we don't want tree nodes are compressed into such kind of isolated core nodes
    remove_node = []
    for node in core_dict:
        neighbors = list(G2.neighbors(node))
        size = len(neighbors)
        count = 0
        for nei in neighbors:
            if nei not in core_dict:
                count += 1
        if count == size:  # if current core node does not have core neighbors
            remove_node.append(node)
    for node in remove_node:
        core_dict.pop(node)

    # sort the core node based on the p-values, and iteratively explore the tree
    # because we want positive tree nodes are compressed into positive core node first
    sorted_core_list = sorted(core_dict, key=lambda x: p_values_dict[x])
    for node in sorted_core_list:
        neighbors = list(G2.neighbors(node))
        children = []
        for nei in neighbors:
            # check if the core_node has positive neighboring tree nodes
            if nei not in core_dict and nei not in tree_visited and G2.nodes[nei]['p'] <= alpha:
                children.append(nei)
        if len(children) != 0:  # if a core node connect to positive tree nodes
            root = node
            roots.append(root)  # means current core node has significant tree nodes connected
            T_subgraph, tree_visited, T_positive = find_tree_subgraph(
                G2, core_dict, tree_visited, root, alpha)

            # merge all connected positive tree nodes into a single core node
            for n in T_positive:
                if n != root:
                    T_subgraph.nodes[root]['pos'] += T_subgraph.nodes[n]['pos']
                    T_subgraph.nodes[root]['neg'] += T_subgraph.nodes[n]['neg']
                    T_subgraph.nodes[root]['p'] = min(T_subgraph.nodes[root]['p'],
                                                      T_subgraph.nodes[n]['p'])
            p_value = T_subgraph.nodes[root]['p']
            pos_list = T_subgraph.nodes[root]['pos']
            neg_list = T_subgraph.nodes[root]['neg']
            # replace current root node with updated attributes
            # G.add_node(root, p=p_value, pos=pos_list, neg=neg_list)
            G.nodes[root]['p'] = p_value
            G.nodes[root]['pos'] = pos_list
            G.nodes[root]['neg'] = neg_list
            pos_count += len(T_subgraph.nodes[root]['pos'])
            neg_count += len(T_subgraph.nodes[root]['neg'])
    # get the compressed core
    nodes = [n for n in G.nodes()]
    for n in nodes:
        if n not in core_dict:
            G.remove_node(n)
    CC = G  # here CC nodes may have isolated nodes
    return CC


def get_pure_sig_supernodes_dict(G):
    """
    input is a networkx graph object, output is a dict of positive weight nodes
    """
    pure_sig_supernodes_dict = {}
    for n in G.nodes():
        if len(G.nodes[n]['pos']) >= 1 and + len(G.nodes[n]['neg']) == 0:
            pure_sig_supernodes_dict[n] = 1
    return pure_sig_supernodes_dict


def merge_adjacent_sig(C):
    """
    use DFS and BFS to find and merge the adjacent pure significant nodes in the core;
    If the adjacent node is not pure significant node, do not merge them now. For example,
    some insignificant nodes may contains significant tree nodes.

    :param C: networkx graph object
    :return: a new merged core
    """
    C = nx.Graph(C)  # to unfreeze the networkx graph object
    positive_core_dict = get_pure_sig_supernodes_dict(C)
    positive_components_dict = {}
    visited = {}
    # loop over all pure significant nodes, and get the positive_components_dict
    # where key is supernode(root) id, value is the list of adjacent positive nodes
    for node in positive_core_dict:
        if node not in visited:
            positive_components_dict[node] = []
            visited[node] = 1
            stack = [node]
            while len(stack) != 0:
                neighbors_list = C.neighbors(stack.pop())  # combine with BFS
                for nei in neighbors_list:
                    if nei in positive_core_dict and nei not in visited:
                        visited[nei] = 1
                        positive_components_dict[node].append(nei)
                        stack.append(nei)
    # for each positive component, merge adjacent positive nodes
    for sup in positive_components_dict:
        if len(positive_components_dict[sup]) >= 1:
            for v in positive_components_dict[sup]:
                C.nodes[sup]['pos'] += C.nodes[v]['pos']
                C.nodes[sup]['neg'] += C.nodes[v]['neg']
                C.nodes[sup]['p'] = min([C.nodes[sup]['p'], C.nodes[v]['p']])
                neg_nei = C.nodes[sup]['neg_nei']
                C.nodes[sup]['neg_nei'] = neg_nei.union(C.nodes[v]['neg_nei'])
                neighbors = list(C.neighbors(v))
                edgelist = [(sup, nei) for nei in neighbors if nei != sup]
                for nei in neighbors:
                    C.nodes[nei]['pos_nei'].discard(v)
                    if nei != sup:
                        C.nodes[nei]['pos_nei'].add(sup)
                C.remove_node(v)
                C.add_edges_from(edgelist)
    return C


def get_ratio_list(C):
    # get a sorted ratio dict for all supernodes
    # the sort is based on the two criteria: ratio and size 
    ratio_dict = {}
    for n in C.nodes():
        N_alpha = len(C.nodes[n]['pos'])
        N = len(C.nodes[n]['pos']) + len(C.nodes[n]['neg'])
        if N_alpha >= 1:
            ratio = float(N_alpha) / N
            ratio_dict[n] = [ratio, N]
    # sort the dict and store in the ordered dict data structure
    # sort the list of items based on the value of dict, which is a list,
    # based on first element first, and then second element
    ordered_ratio_dict = collections.OrderedDict(
        sorted(ratio_dict.items(), key=lambda t: t[1], reverse=True))
    return ordered_ratio_dict


def get_significant_dict(G):
    """
    input is a networkx graph object, output is a dict of significant nodes
    """
    sig_dict = {}
    for n in G.nodes():
        if len(G.nodes[n]['pos']) > 0:
            sig_dict[n] = 1
    return sig_dict


def highest_ratio_detection(C, ordered_ratio_dict, seed=None):
    """
    apply 3 methods to find highest ratio among all supernodes
    to find the best cluster, and return the target_supernode
    and target_method which lead to highest ratio

    Note, we should not change the core C here. Because in this step,
    we are still detecting, not doing the merge yet.
    """

    highest_ratio = 0
    target_supernode = -1
    target_method = -1
    merged_neighbor = -1
    for n in ordered_ratio_dict:
        current_highest_ratio = ordered_ratio_dict[n][0]  # get the ratio of current super node
        if current_highest_ratio < highest_ratio:
            # if the ratio of current supernode is
            # less than the highest ratio, then break and
            # omit following procedures, because ratio after merge is always
            # less than the highest ratio.
            break
        else:
            ratios = []
            merged = []
            pos_number0 = len(C.nodes[n]['pos'])
            neg_number0 = len(C.nodes[n]['neg'])
            # neighbors_list = list(C.neighbors(n))  # get current supernode's neighbors

            # method 1: merge the two non-pure significant supernodes if they are adjacent
            # size = len(list(neighbors_list))
            if len(C.nodes[n]['pos_nei']) == 0:
                ratio1 = 0
                ratios.append(ratio1)
                merged.append(None)
            else:
                for nei in C.nodes[n]['pos_nei']:
                    if nei != n:
                        try:
                            pos_number1 = len(C.nodes[nei]['pos'])
                            neg_number1 = len(C.nodes[nei]['neg'])
                        except:
                            print(seed)
                        pos1 = pos_number0 + pos_number1
                        neg1 = neg_number0 + neg_number1
                        ratio1 = float(pos1) / (pos1 + neg1)
                        ratios.append(ratio1)
                        merged.append(nei)
                        break

            # method 2: merge two significant nodes if they are one node away
            if len(C.nodes[n]['neg_nei']) == 0:
                ratio2 = 0.
                ratios.append(ratio2)
                merged.append(None)
            else:
                for nei in C.nodes[n]['neg_nei']:
                    nei_pos_nei_size = len(C.nodes[nei]['pos_nei'])
                    pos_number2 = len(C.nodes[nei]['pos'])
                    neg_number2 = len(C.nodes[nei]['neg'])
                    pos2 = pos_number2 + pos_number0
                    neg2 = neg_number2 + neg_number0
                    if nei_pos_nei_size >= 2:  # except current n
                        ratio2 = float(pos2) / (pos2 + neg2)
                        ratios.append(ratio2)
                        merged.append(nei)
                        break

            # method 3: merge the non-significant node with highest degree to
            # the highest ratio super node
            if len(C.nodes[n]['neg_nei']) == 0:
                ratio3 = 0.
                ratios.append(ratio3)
                merged.append(None)
            else:
                max_deg = 0
                max_nei = None
                for nei in C.nodes[n]['neg_nei']:
                    deg = C.degree[nei]
                    if deg > max_deg:
                        max_deg = deg
                        max_nei = nei
                neg_number3 = len(C.nodes[max_nei]['neg'])
                ratio3 = float(pos_number0) / (pos_number0 + neg_number0 + neg_number3)
                ratios.append(ratio3)
                merged.append(max_nei)

            # print 'ratios',ratios
            max_ratio = max(ratios)  # find the highest ratio among three methods
            index = np.argmax(ratios)
            best_neighbor = merged[index]
            if max_ratio > highest_ratio:  # if higher, then merge and update the ordered
                highest_ratio = max_ratio
                target_supernode = n
                target_method = index + 1
                merged_neighbor = best_neighbor
    return highest_ratio, target_supernode, target_method, merged_neighbor


def apply_merge(C, highest_ratio, target_supernode, merged_neighbor):
    """
    Given the target supernode and target method, we can apply the merge,
    and update the ordered ratio dict

    :param C: networkx graph object
    :param ordered_ratio_dict:
    :param highest_ratio:
    :param target_supernode:
    :param target_method:
    :return: the core after one merge
    """
    C = nx.Graph(C)
    # print(target_supernode, C.nodes[target_supernode])
    # print(merged_neighbor, C.nodes[merged_neighbor])
    C.nodes[target_supernode]['pos'] += C.nodes[merged_neighbor]['pos']
    C.nodes[target_supernode]['neg'] += C.nodes[merged_neighbor]['neg']
    pos_size = len(C.nodes[target_supernode]['pos'])
    neg_size = len(C.nodes[target_supernode]['neg'])
    ratio = float(pos_size) / (pos_size + neg_size)
    assert highest_ratio == ratio
    C.nodes[target_supernode]['p'] = min(
        [C.nodes[target_supernode]['p'], C.nodes[merged_neighbor]['p']])
    pos_nei = C.nodes[target_supernode]['pos_nei']
    neg_nei = C.nodes[target_supernode]['neg_nei']
    C.nodes[target_supernode]['pos_nei'] = pos_nei.union(C.nodes[merged_neighbor]['pos_nei'])
    C.nodes[target_supernode]['neg_nei'] = neg_nei.union(C.nodes[merged_neighbor]['neg_nei'])
    neighbors = list(C.neighbors(merged_neighbor))
    for nei in neighbors:
        C.nodes[nei]['pos_nei'].discard(merged_neighbor)
        C.nodes[nei]['neg_nei'].discard(merged_neighbor)
        if nei != target_supernode:
            C.nodes[nei]['pos_nei'].add(target_supernode)
    edgelist = [(target_supernode, neigh) for neigh in neighbors if neigh != target_supernode]
    C.remove_node(merged_neighbor)
    C.add_edges_from(edgelist)
    for n in C.nodes():
        C.nodes[n]['neg_nei'].discard(merged_neighbor)
        C.nodes[n]['pos_nei'].discard(merged_neighbor)
    return C


def merge_and_record(G, seed=None, save_subgraph=False):
    # CC = G.subgraph(max(nx.connected_components(G), key=len))
    # CC = preprocess_graph(CC, alpha)
    CC = merge_adjacent_sig(G)

    # start merge from the largest pure significant super node
    ordered_ratio_dict = get_ratio_list(CC)
    if len(ordered_ratio_dict.keys()) == 0:  # if no significant node
        return {(0, 0): []}
    top_node = list(ordered_ratio_dict.keys())[0]
    N_alpha = len(CC.nodes[top_node]['pos'])
    N = N_alpha + len(CC.nodes[top_node]['neg'])

    if save_subgraph is True:
        # if we are running on single real graph, then we try to store all 
        # subgraphs for all (N, N_alpha) pairs
        significant_subgraph = []
        significant_subgraph += CC.nodes[top_node]['pos']
        significant_subgraph += CC.nodes[top_node]['neg']
        records = {(N, N_alpha): significant_subgraph}
        while len(ordered_ratio_dict) != 1:  # need to think about the condition more carefully
            highest_ratio, target_supernode, target_method, merged_neighbor = highest_ratio_detection(
                CC, ordered_ratio_dict, seed)
            CC = apply_merge(CC, highest_ratio, target_supernode, merged_neighbor)
            N_alpha = len(CC.nodes[target_supernode]['pos'])
            N = N_alpha + len(CC.nodes[target_supernode]['neg'])
            significant_subgraph = []
            significant_subgraph += CC.nodes[target_supernode]['pos']
            significant_subgraph += CC.nodes[target_supernode]['neg']
            if target_method == 1:
                if (N, N_alpha) not in records:
                    records[(N, N_alpha)] = significant_subgraph
            ordered_ratio_dict = get_ratio_list(CC)
    else:
        # if case name is not none, then we need to store the (N, N_alpha) pairs 
        # without corresponding subgraphs
        records = {(N, N_alpha): True}
        while len(ordered_ratio_dict) != 1:  # need to think about the condition more carefully
            highest_ratio, target_supernode, target_method, merged_neighbor = highest_ratio_detection(
                CC, ordered_ratio_dict, seed)
            # print(highest_ratio, target_supernode, target_method, merged_neighbor)
            CC = apply_merge(CC, highest_ratio, target_supernode, merged_neighbor)
            N_alpha = len(CC.nodes[target_supernode]['pos'])
            N = N_alpha + len(CC.nodes[target_supernode]['neg'])
            if target_method == 1:
                if (N, N_alpha) not in records:
                    records[(N, N_alpha)] = True
            ordered_ratio_dict = get_ratio_list(CC)

    # sort the (N, N_alpha) pairs by N, and remove non-optimal ratio pairs
    records_keys = sorted(records.keys(), reverse=True)
    current_ratio = float(records_keys[0][1]) / records_keys[0][0]
    records_copy = [e for e in records_keys]
    for (N, N_alpha) in records_copy[1:]:
        ratio = float(N_alpha) / N
        if ratio >= current_ratio:
            current_ratio = ratio
        else:
            records_keys.remove((N, N_alpha))
            del records[(N, N_alpha)]

    return records


def preprocess_graph(G, alpha, C=None):
    for v in G.nodes():
        p_val = G.nodes[v]['p']
        if p_val <= alpha:
            G.nodes[v]['pos'] = [v]
            G.nodes[v]['neg'] = []
            G.nodes[v]['pos_nei'] = set()
            G.nodes[v]['neg_nei'] = set()
        else:
            G.nodes[v]['pos'] = []
            G.nodes[v]['neg'] = [v]
            G.nodes[v]['pos_nei'] = set()
            G.nodes[v]['neg_nei'] = set()

    if C:  # if core tree decomposition then tree compression
        G = compress(G, C, alpha)
    for (v1, v2) in G.edges():
        v1_p_val = G.nodes[v1]['p']
        v2_p_val = G.nodes[v2]['p']
        if v1_p_val <= alpha:
            G.nodes[v2]['pos_nei'].add(v1)
        else:
            G.nodes[v2]['neg_nei'].add(v1)
        if v2_p_val <= alpha:
            G.nodes[v1]['pos_nei'].add(v2)
        else:
            G.nodes[v1]['neg_nei'].add(v2)

    return G
