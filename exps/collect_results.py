import json


def collect(graph_name, alter_type, method, ctd, case_id_list):
    result_dir = "results/{}".format(graph_name)
    if ctd:
        output_dir = "{}/exp_outputs/{}/{}_ctd".format(result_dir, alter_type, method)
    else:
        output_dir = "{}/exp_outputs/{}/{}".format(result_dir, alter_type, method)

    for case_id in case_id_list:
        with open("{}/{}.json".format(output_dir, case_id), "r", encoding="utf-8") as f:
            result_dict = json.load(f)
            print("{},{},{},{},{},{},{},{},{},{},{},{}".format(
                result_dict["graph_name"],
                result_dict["alter_type"],
                result_dict["method"],
                result_dict["core_tree_decomposition"],
                result_dict["case_id"],
                result_dict["N"],
                result_dict["alpha"],
                result_dict["N_alpha"],
                result_dict["alpha_prime"],
                result_dict["precision"],
                result_dict["recall"],
                result_dict["f1"]
            ))


if __name__ == '__main__':
    graph_name = 'wikivote'
    alter_type = "alpha_0.01_signal_5"
    method = "randomization_tests"
    ctd = False
    case_id_list = [i for i in range(1, 50)]
    collect(graph_name, alter_type, method, ctd, case_id_list)
