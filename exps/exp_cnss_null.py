"""
experiments on simulation of null runs in parallel,
we view this procedure as our data pre-processing step
"""

from cnss.cnss import CNSS
import pickle
from cnss.utils import make_dir
import numpy as np

if __name__ == "__main__":
    graph_name = 'wikivote'

    # step 1. obtain alpha-prime for H0
    file_path = "../data/{}/G.pkl".format(graph_name)
    G = pickle.load(open(file_path, "rb"))
    result_dir = "results/{}".format(graph_name)
    make_dir(result_dir)
    alpha_list = [i / 1000.0 for i in range(1, 10)] + [j / 100.0 for j in range(1, 10)]
    cnss = CNSS(G, alpha_list, result_dir, method="lower_bounds")
    # cases_list = list(range(200))
    # cnss.randomization_tests(cases_list=cases_list, num_cpus=60)
    cnss.neighbor_analysis()
