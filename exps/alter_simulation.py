import pickle
import networkx as nx
import numpy as np
import random
from multiprocessing import Pool
from cnss.utils import make_dir


def random_walk(G, size, random_state):
    np.random.seed(random_state)
    random.seed(random_state)
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


def alter_piecewise_uniform_simulation(graph_name, true_size, alpha, signal_strength, case_id):
    np.random.seed(case_id)
    random.seed(case_id)
    G = pickle.load(open("../data/{}/G.pkl".format(graph_name), "rb"))
    true_subgraph = random_walk(G, true_size, random_state=case_id)
    p_values_dict = {}
    sig_true = 0.
    insig_true = 0.
    sig_not_true = 0.
    insig_not_true = 0.
    for node in G.nodes():
        if node not in true_subgraph:
            p_val = np.random.uniform(0., 1.)
            if p_val <= alpha:
                sig_not_true += 1
            else:
                insig_not_true += 1
        else:
            prob = signal_strength * alpha
            cho = np.random.choice([0, 1], p=[1. - prob, prob])
            if cho:
                p_val = np.random.uniform(0., alpha)
                # print("significant: {}".format(p_val))
            else:
                p_val = np.random.uniform(alpha, 1.)
            if p_val <= alpha:
                sig_true += 1
            else:
                insig_true += 1
        p_values_dict[node] = p_val

    output_dir = "../data/{}/simulations/alpha_{}_signal_{}".format(
        graph_name, alpha, signal_strength)
    make_dir(output_dir)
    pickle.dump(true_subgraph, open("{}/{}_ground_truth_subgraph.pkl".format(
        output_dir, case_id), "wb"))
    pickle.dump(p_values_dict, open("{}/{}_p_values_dict.pkl".format(
        output_dir, case_id), "wb"))
    print(case_id, sig_true, insig_true, sig_not_true, insig_not_true, signal_strength * alpha,
          (sig_true + sig_not_true) / G.number_of_nodes())


if __name__ == '__main__':
    graph_name = "wikivote"
    true_size = 100
    alpha = 0.01
    para_list = []
    for signal_strength in [10, 25, 50, 75, 100]:
        for case_id in range(50):
            para = (graph_name, true_size, alpha, signal_strength, case_id)
            para_list.append(para)
            alter_piecewise_uniform_simulation(*para)
    # pool = Pool(processes=50)
    # pool.starmap(alter_piecewise_uniform_simulation, para_list)
    # pool.close()
    # pool.join()
