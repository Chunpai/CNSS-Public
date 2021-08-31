import json

from cnss.cnss import CNSS
import pickle
from cnss.utils import make_dir
import numpy as np
from multiprocessing import Pool


def evaluate(detected_subgraph, true_subgraph):
    """
    evaluate the detected subgraph
    :param true_subgraph
    :param detected_subgraph: a list of detected nodes
    :return: precision, recall, and f1-score
    """
    detected_subgraph = set(detected_subgraph)
    true_subgraph = set(true_subgraph)
    union = true_subgraph.union(detected_subgraph)
    intersect = true_subgraph.intersection(detected_subgraph)
    prec = float(len(intersect)) / len(detected_subgraph)
    recall = float(len(intersect)) / len(true_subgraph)
    if prec == 0 and recall == 0:
        f1 = 0.
    else:
        f1 = 2 * prec * recall / (prec + recall)
    print("Precision: {}".format(prec))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    return prec, recall, f1


def run_single_case(graph_name, alter_type, case_id, method="lower_bounds", ctd=False, exp=False):
    """
    @param graph_name:
    @param alter_type: Gaussian or Piecewise uniform
    @param case_id: case id of alternative hypothesis simulation
    @param method: calibration method or uncalibrated
    @param ctd: user core tree decomposition or not
    @param exp: running multiple experiments or single case
    @return:
    """
    file_path = "../data/{}/G.pkl".format(graph_name)
    G = pickle.load(open(file_path, "rb"))
    file_path = "../data/{}/simulations/{}/{}_p_values_dict.pkl".format(
        graph_name, alter_type, case_id)
    p_values_dict = pickle.load(open(file_path, "rb"))
    file_path = "../data/{}/simulations/{}/{}_ground_truth_subgraph.pkl".format(
        graph_name, alter_type, case_id)
    true_subgraph = pickle.load(open(file_path, "rb"))
    true_subgraph = true_subgraph.keys()
    for n in G.nodes():
        G.nodes[n]['p'] = p_values_dict[n]
    if ctd:
        try:
            file_path = "../data/{}/C.pkl".format(graph_name)
            C = pickle.load(open(file_path, "rb"))
        except (EOFError, FileNotFoundError) as e:
            C = None
    else:
        C = None

    result_dir = "results/{}".format(graph_name)
    make_dir(result_dir)
    alpha_list = [i / 1000.0 for i in range(1, 10)] + [j / 100.0 for j in range(1, 10)]
    cnss = CNSS(G, alpha_list, result_dir, method=method, C=C)
    cnss.detect(seed=case_id, save_subgraph=True)
    print("subgraph: {}".format(cnss.optimal_S))
    print("N: {}".format(cnss.optimal_N))
    print("N_alpha: {}".format(cnss.optimal_N_alpha))
    print("alpha: {}".format(cnss.optimal_alpha))
    print("alpha prime: {}".format(cnss.optimal_alpha_prime))
    prec, recall, f1 = evaluate(cnss.optimal_S, true_subgraph)
    if run_exp:
        if ctd:
            output_dir = "{}/exp_outputs/{}/{}_ctd".format(result_dir, alter_type, method)
        else:
            output_dir = "{}/exp_outputs/{}/{}".format(result_dir, alter_type, method)
        make_dir(output_dir)
        out_dict = {
            "graph_name": graph_name,
            "alter_type": alter_type,
            "method": method,
            "core_tree_decomposition": ctd,
            "case_id": case_id,
            "N": cnss.optimal_N,
            "N_alpha": cnss.optimal_N_alpha,
            "alpha": cnss.optimal_alpha,
            "alpha_prime": cnss.optimal_alpha_prime,
            "precision": prec,
            "recall": recall,
            "f1": f1,
            "detected_subgraph": sorted(list(set(cnss.optimal_S))),
            "true_subgraph": sorted(list(set(true_subgraph)))
        }
        with open("{}/{}.json".format(output_dir, case_id), "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)


def run_exp(graph_name, alter_type, method, ctd=False, num_cpus=50):
    """
    @param graph_name:
    @param alter_type:
    @return:
    """
    exp = True
    para_list = []
    for case_id in range(50):
        para = (graph_name, alter_type, case_id, method, ctd, exp)
        para_list.append(para)
    pool = Pool(processes=50)
    pool.starmap(run_single_case, para_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    # a demo to run the CNSS on a single case
    graph_name = 'wikivote'
    # alter_type = "mu1.5"
    alter_type = "alpha_0.01_signal_5"
    # case_id = 0
    # run_single_case(graph_name, alter_type, case_id)

    run_exp(graph_name, alter_type, method="randomization_tests", num_cpus=25)
    # run_exp(graph_name, alter_type, method="lower_bounds", num_cpus=25)
    # run_exp(graph_name, alter_type, method="uncalibrated", num_cpus=25)

    # run_exp(graph_name, alter_type, method="uncalibrated", ctd=True, num_cpus=25)
    # run_exp(graph_name, alter_type, method="lower_bounds", ctd=True, num_cpus=25)
    # run_exp(graph_name, alter_type, method="randomization_tests", ctd=True, num_cpus=25)
