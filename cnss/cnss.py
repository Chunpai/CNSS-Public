import copy
import pickle

from cnss.merge import merge_and_record, preprocess_graph
from cnss.utils import berk_jones_scan_statistic, make_dir
import numpy as np
from multiprocessing import Pool
from os import path
from scipy.special import lambertw
import networkx as nx
import time


class CNSS(object):
    def __init__(self, G, alpha_list, result_dir, methods=None, C=None, results_dict=None,
                 **kwargs):
        """
        G is the networkx graph object with node attribute "p" to store p-value.
        C is the networkx graph object that contains the core of graph G, if C is not None, then
            we apply the CNSS with core tree decomposition
        method: calibration method or uncalibrated
        """
        self.result_dir = result_dir
        self.G = G
        self.C = C
        self.average_degree = np.mean([self.G.degree[v] for v in self.G.nodes()])
        self.alpha_list = alpha_list
        self.alpha_time_dict = {}

        self.methods = methods
        for method in self.methods:
            if method == "lower_bounds" or method == "neighbor_analysis":
                file_path = "{}/alpha_prime/neighbor_analysis_alpha_prime.pkl".format(
                    self.result_dir)
                if path.exists(file_path):
                    self.neighbor_analysis_alpha_prime = pickle.load(open(file_path, "rb"))
            elif method == "randomization_tests":
                file_path = "{}/alpha_prime/randomization_tests_alpha_prime.pkl".format(
                    self.result_dir)
                if path.exists(file_path):
                    self.randomization_tests_alpha_prime = pickle.load(open(file_path, "rb"))
            elif method == "uncalibrated" or method == "percolation_theory":
                # uncalibrated CNSS or use percolation theory as lower bound for calibration
                pass
            else:
                raise ValueError

        # if previous searching result exists, load previous results and avoid repeat searching
        if results_dict is not None:
            # notice that results_dict could be {} as well
            self.detection_results_dict = results_dict
            # remove the alpha from alpha_list which had been searched before
            for method in self.detection_results_dict.keys():
                self.alpha_time_dict = self.detection_results_dict[method]["alpha_time_dict"]
                old_alpha_list = list(self.alpha_time_dict.keys())
                self.alpha_list = list(set(alpha_list) - set(old_alpha_list))
                break
        else:
            self.detection_results_dict = {}

    def detect(self, seed=None, save_subgraph=False):
        """
        detect anomalous subgraph
        @param seed:
        @param save_subgraph:
        @return:
        """
        print("alpha list: {}".format(self.alpha_list))
        G = self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()  # need copy()
        for alpha in self.alpha_list:
            start_time = time.time()
            print("current significance threshold alpha: {}".format(alpha))
            new_G = copy.deepcopy(G)
            new_G = preprocess_graph(new_G, alpha, self.C)
            candidate_results = merge_and_record(new_G, seed, save_subgraph)
            for N, N_alpha in candidate_results:
                if save_subgraph is True:
                    S = candidate_results[(N, N_alpha)]
                else:
                    # NotImplemented
                    S = None
                for method in self.methods:
                    if method not in self.detection_results_dict:
                        self.detection_results_dict[method] = {"max_score": 0}
                    alpha_prime = self.get_alpha_prime(N, alpha, method)
                    if alpha_prime <= 0.:  # percolation theory may get non-positive
                        continue
                    calibrated_bj_score = berk_jones_scan_statistic(N, N_alpha, alpha_prime)
                    if calibrated_bj_score >= self.detection_results_dict[method]["max_score"]:
                        self.detection_results_dict[method]["max_score"] = calibrated_bj_score
                        self.detection_results_dict[method]["optimal_S"] = S
                        self.detection_results_dict[method]["optimal_N_alpha"] = N_alpha
                        self.detection_results_dict[method]["optimal_N"] = N
                        self.detection_results_dict[method]["optimal_alpha"] = alpha
                        self.detection_results_dict[method]["optimal_alpha_prime"] = alpha_prime
            run_time = time.time() - start_time
            self.alpha_time_dict[alpha] = run_time

    def get_alpha_prime(self, N, alpha, method):
        """
        loading the alpha-prime for N_alpha and N, which was pre-computed based on
            randomization tests, neighbor analysis, or percolation theory
        para: N
        para: N_alpha
        para: method could be 'randomization' or 'lower_bounds'
            for 'lower_bounds' we use the alpha_prime = max(alpha_prime_1, alpha_prime_2)
            where alpha_prime_1 denotes the lower bound obtained from neighbor analysis
            and alpha_prime_2 denotes the lower bound obtained from percolation theory
        """
        if method == "neighbor_analysis":
            alpha_prime = self.neighbor_analysis_alpha_prime[alpha][N]
        elif method == "percolation_theory":
            k = self.average_degree * 2 - 1e-5
            p_infty = (lambertw(-alpha * np.exp(-alpha * k) * k) + alpha * k) / k
            if N < self.G.number_of_nodes() * p_infty.real:
                alpha_prime = 1
            else:
                alpha_prime = float(self.G.number_of_nodes() * p_infty.real) / N
        elif method == "lower_bounds":
            neighbor_analysis_alpha_prime = self.neighbor_analysis_alpha_prime[alpha][N]
            k = self.average_degree * 2 - 1e-5
            p_infty = (lambertw(-alpha * np.exp(-alpha * k) * k) + alpha * k) / k
            if N < self.G.number_of_nodes() * p_infty.real:
                percolation_theory_alpha_prime = 1
            else:
                percolation_theory_alpha_prime = float(self.G.number_of_nodes() * p_infty.real) / N
            alpha_prime = max(neighbor_analysis_alpha_prime, percolation_theory_alpha_prime)
        elif method == "randomization_tests":
            alpha_prime = self.randomization_tests_alpha_prime[(N, alpha)]
        elif method == "uncalibrated":
            alpha_prime = alpha
        else:
            alpha_prime = None
        return alpha_prime

    def randomization_tests(self, cases_list, num_cpus=1):
        """
        for null hypothesis preprocessing step
        @param cases_list:
        @param num_cpus:
        @return:
        """
        para_list = []
        pool_results = []
        for case_id in cases_list:
            # simulate p-values for null hypothesis
            np.random.seed(case_id)
            G = copy.deepcopy(self.G)
            for v in G.nodes():
                p_val = np.random.uniform(0, 1)
                G.nodes[v]['p'] = p_val
            CC = G.subgraph(max(nx.connected_components(G), key=len))
            for alpha in self.alpha_list:
                new_G = preprocess_graph(CC, alpha)
                para = (new_G, case_id, False)
                para_list.append(para)
        if num_cpus > 1:
            pool = Pool(processes=num_cpus)
            pool_results = pool.starmap(merge_and_record, para_list)
            pool.close()
            pool.join()
        else:
            for para in para_list:
                print(para)
                result = merge_and_record(*para)
                print("number of detected pairs: {}".format(len(result)))
                pool_results.append(result)

    def collect_alpha_prime(self):
        randomization_tests_results = {}
        for index, result in enumerate(pool_results):
            alpha = para_list[index][1]  # TODO
            N_map_N_alpha = self.interpolation(result)
            for N in N_map_N_alpha.keys():
                N_alpha = N_map_N_alpha[N]["N_alpha"]
                if (N, alpha) not in randomization_tests_results:
                    randomization_tests_results[(N, alpha)] = []
                randomization_tests_results[(N, alpha)].append(N_alpha)
        file_name = "{}/alpha_prime/randomization_tests.pkl".format(self.result_dir)
        pickle.dump(randomization_tests_results, open(file_name, "wb"))

        randomization_tests_alpha_prime = {}
        for (N, alpha) in randomization_tests_results:
            alpha_prime = max(randomization_tests_results[(N, alpha)]) / N
            randomization_tests_alpha_prime[(N, alpha)] = alpha_prime
        file_name = "{}/alpha_prime/randomization_tests_alpha_prime.pkl".format(self.result_dir)
        pickle.dump(randomization_tests_alpha_prime, open(file_name, "wb"))

    def interpolation(self, N_and_N_alpha_dict):
        """
        for null hypothesis preprocessing step
        @param N_and_N_alpha_dict:
        @return:
        """
        num_nodes = self.G.number_of_nodes()
        mapping = {}
        # in mapping, keys are detected N, and for each detected N, it has keys
        #   "upper_N", "lower_N", and "N_alpha"
        # for each non-detected N, it only has key "N_alpha" which is estimated
        # based on interpolation
        pairs = sorted(list(N_and_N_alpha_dict.keys()), reverse=True)
        for index, (N, N_alpha) in enumerate(pairs):
            mapping[N] = {}
            mapping[N]["N_alpha"] = N_alpha
            if index == 0:
                mapping[N]["upper_N"] = num_nodes
                mapping[N]["lower_N"] = pairs[index + 1][0]
                upper_N_alpha = N_alpha
                mapping[num_nodes] = {}
                mapping[num_nodes]["N_alpha"] = upper_N_alpha
                mapping[num_nodes]["upper_N"] = num_nodes
                mapping[num_nodes]["lower_N"] = N
                # upper_N_alpha = N_alpha + (num_nodes - N) * alpha
            elif index == len(pairs) - 1:
                mapping[N]["upper_N"] = pairs[index - 1][0]
                mapping[N]["lower_N"] = N
            else:
                mapping[N]["upper_N"] = pairs[index - 1][0]
                mapping[N]["lower_N"] = pairs[index + 1][0]
        all_detected_N = list(mapping.keys())
        for N in sorted(all_detected_N, reverse=True):
            upper_N = mapping[N]["upper_N"]
            upper_N_alpha = mapping[upper_N]["N_alpha"]
            N_alpha = mapping[N]["N_alpha"]
            if N == N_alpha:
                for n in range(1, N):
                    if n not in mapping:
                        mapping[n] = {}
                    mapping[n]["N_alpha"] = n
                break
            else:
                for n in range(N + 1, upper_N):
                    n_alpha_est = float(upper_N_alpha * n) / upper_N
                    n_alpha_est = max(n_alpha_est, N_alpha)
                    mapping[n] = {}
                    mapping[n]["N_alpha"] = n_alpha_est
        return mapping

    def neighbor_analysis(self):
        """
        for null hypothesis preprocessing step
        @return:
        """
        num_nodes = self.G.number_of_nodes()
        subgraph = {}  # key is node
        subgraph_neighbors = {}  # key is neighbor node, value is p-value
        records = {}
        subgraph_neighbors_list = []
        while len(subgraph) != num_nodes:
            if len(subgraph) == 0:
                # set root node as highest degree node
                root = sorted(self.G.degree, key=lambda x: x[1], reverse=True)[0][0]
                subgraph[root] = True
                added_new_node = root
                for n in self.G.neighbors(root):
                    subgraph_neighbors[n] = True
                    subgraph_neighbors_list.append(n)
                records[(len(subgraph), len(subgraph_neighbors_list))] = (
                    added_new_node, subgraph_neighbors_list, [])
            else:
                neighbor_out_degree_dict = {}
                # find next neighbor
                # print("time 1: {}".format(time.time() - start_time))
                for nei in subgraph_neighbors.keys():
                    # compute new out-degree of each neighbor nei if we include it into subgraph
                    new_out_degree = 0
                    for n in self.G.neighbors(nei):
                        if n not in subgraph:
                            new_out_degree += 1
                    neighbor_out_degree_dict[nei] = new_out_degree
                next_node = \
                    sorted(neighbor_out_degree_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
                # print("time 2: {}".format(time.time() - start_time))
                # pick the one with highest neighbor out degree
                subgraph[next_node] = True
                added_new_node = next_node
                subgraph_neighbors.pop(next_node)
                removed_subgraph_neighbors = [next_node]
                added_subgraph_neighbors = []
                for n in self.G.neighbors(next_node):
                    if n not in subgraph:
                        subgraph_neighbors[n] = True
                        added_subgraph_neighbors.append(n)
                # print("time 3: {}".format(time.time() - start_time))
                records[(len(subgraph), len(subgraph_neighbors))] = (
                    added_new_node, added_subgraph_neighbors, removed_subgraph_neighbors)
        file_name = "{}/alpha_prime/neighbor_analysis.pkl".format(self.result_dir)
        pickle.dump(records, open(file_name, "wb"))

        neighbor_analysis_alpha_prime = {}
        for alpha in self.alpha_list:
            alpha_prime_dict = {1: 1.}
            for c, kc in sorted(records.keys()):
                for n in range(int(c), int(c + kc) + 1):
                    alpha_prime = float(c * alpha + min(kc * alpha, n - c)) / n
                    if n not in alpha_prime_dict:
                        alpha_prime_dict[n] = alpha_prime
                    else:
                        if alpha > alpha_prime_dict[n]:
                            alpha_prime_dict[n] = alpha_prime

            # remove sub-optimum alpha-prime
            for n in sorted(alpha_prime_dict.keys(), reverse=True):
                alpha_prime = alpha_prime_dict[n]
                if n - 1 in alpha_prime_dict:
                    if alpha_prime > alpha_prime_dict[n - 1]:
                        alpha_prime_dict[n - 1] = alpha_prime
            neighbor_analysis_alpha_prime[alpha] = copy.deepcopy(alpha_prime_dict)
        file_name = "{}/alpha_prime/neighbor_analysis_alpha_prime.pkl".format(self.result_dir)
        pickle.dump(neighbor_analysis_alpha_prime, open(file_name, "wb"))
