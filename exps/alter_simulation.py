import pickle
import networkx as nx
import numpy as np
import random
from multiprocessing import Pool
from cnss.utils import make_dir


def random_walk(G, size, random_state):
    np.random.seed(random_state)
    length = G.number_of_nodes()
    current_vertex = np.random.randint(0, length)  # choose a random starting node
    ground_truth_subgraph = {current_vertex: 1}
    assert size < length
    while len(ground_truth_subgraph) < size:
        neighbors = list(G.neighbors(current_vertex))
        next_id = random.choice(neighbors)
        if next_id not in ground_truth_subgraph:
            ground_truth_subgraph[next_id] = 1
        current_vertex = next_id
    return ground_truth_subgraph


def alter_piecewise_uniform_simulation(graph_name, alpha, signal_strength, case_id):
    np.random.seed(case_id)
    G = pickle.load(open("../data/{}/G.pkl".format(graph_name), "rb"))
    num_nodes = G.number_of_nodes()
    num_sig_nodes = num_nodes * alpha * signal_strength
    true_subgraph = random_walk(G, num_sig_nodes, random_state=case_id)
    p_values_dict = {}
    for node in G.nodes():
        if node in true_subgraph:
            p_val = np.random.uniform(0., alpha)
        else:
            p_val = np.random.uniform(alpha, 1.)
        p_values_dict[node] = p_val

    output_dir = "../data/{}/simulations/alpha_{}_signal_{}".format(
        graph_name, alpha, signal_strength)
    make_dir(output_dir)
    pickle.dump(true_subgraph, open("{}/{}_ground_truth_subgraph.pkl".format(
        output_dir, case_id), "wb"))
    pickle.dump(p_values_dict, open("{}/{}_p_values_dict.pkl".format(
        output_dir, case_id), "wb"))
    # return true_subgraph, p_values_dict
    print(case_id)
    print("subgraph", true_subgraph)
    print("p values", p_values_dict)


if __name__ == '__main__':
    graph_name = "wikivote"
    alpha = 0.01
    para_list = []
    for signal_strength in [2, 3, 4, 5]:
        for case_id in range(50):
            para = (graph_name, alpha, signal_strength, case_id)
            para_list.append(para)
            # alter_piecewise_uniform_simulation(*para)
    pool = Pool(processes=60)
    pool.starmap(alter_piecewise_uniform_simulation, para_list)
    pool.close()
    pool.join()
