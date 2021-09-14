"""
experiments on simulation of null runs in parallel,
we view this procedure as our data pre-processing step
"""

from cnss.cnss import CNSS
import pickle
from cnss.utils import make_dir
import numpy as np

if __name__ == "__main__":
    # graph_name = 'wikivote'
    graph_name = 'condmat'

    # step 1. obtain alpha-prime for H0
    file_path = "../data/{}/G.pkl".format(graph_name)
    G = pickle.load(open(file_path, "rb"))
    # file_path = "../data/{}/C.pkl".format(graph_name)
    # C = pickle.load(open(file_path, "rb"))
    # file_path = "../data/{}/simulations/mu1.5/0_p_values_dict.pkl".format(graph_name)
    # p_values_dict = pickle.load(open(file_path, "rb"))
    # core_dict = dict([(n, 1) for n in C.nodes()])
    # sorted_core_list = sorted(core_dict.keys(), key=lambda x: p_values_dict[x])
    # for n in sorted_core_list:
    #     print(n, p_values_dict[n])
    # print(sorted_core_list)
    result_dir = "results/{}".format(graph_name)
    make_dir(result_dir)
    alpha_list = [i / 1000.0 for i in range(1, 10)] + [j / 100.0 for j in range(1, 10)]
    cnss = CNSS(G, alpha_list, result_dir, methods=["randomization_tests"])
    cases_list = list(range(200))
    cnss.randomization_tests(cases_list=cases_list, num_cpus=50)
    # cnss.neighbor_analysis()
