import os
import pathlib
import logging
import numpy as np

strBold = lambda skk: "\033[1m {}\033[00m".format(skk)
strBlue = lambda skk: "\033[34m {}\033[00m".format(skk)
strRed = lambda skk: "\033[91m {}\033[00m".format(skk)
strGreen = lambda skk: "\033[92m {}\033[00m".format(skk)
strYellow = lambda skk: "\033[93m {}\033[00m".format(skk)
strLightPurple = lambda skk: "\033[94m {}\033[00m".format(skk)
strPurple = lambda skk: "\033[95m {}\033[00m".format(skk)
strCyan = lambda skk: "\033[96m {}\033[00m".format(skk)
strLightGray = lambda skk: "\033[97m {}\033[00m".format(skk)
strBlack = lambda skk: "\033[98m {}\033[00m".format(skk)

prBold = lambda skk: print("\033[1m {}\033[00m".format(skk))
prBlue = lambda skk: print("\033[34m {}\033[00m".format(skk))
prRed = lambda skk: print("\033[91m {}\033[00m".format(skk))
prGreen = lambda skk: print("\033[92m {}\033[00m".format(skk))
prYellow = lambda skk: print("\033[93m {}\033[00m".format(skk))
prLightPurple = lambda skk: print("\033[94m {}\033[00m".format(skk))
prPurple = lambda skk: print("\033[95m {}\033[00m".format(skk))
prCyan = lambda skk: print("\033[96m {}\033[00m".format(skk))
prLightGray = lambda skk: print("\033[97m {}\033[00m".format(skk))
prBlack = lambda skk: print("\033[98m {}\033[00m".format(skk))


def create_logger(log_file):
    """
    create logger for different purpose
    :param log_file: the place to store the log
    :return:
    """
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("[%(asctime)s-%(levelname)s-%(name)s]:%(message)s")
    formatter = logging.Formatter("[%(asctime)s-%(levelname)s]:%(message)s")

    if log_file is not None:
        file_handler = logging.FileHandler("{}.log".format(log_file))
        file_handler.setLevel(logging.INFO)  # only INFO in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # show DEBUG on console
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def berk_jones_scan_statistic(N, N_alpha, alpha):
    significance_ratio = N_alpha * 1. / N
    return N * 1. * KL(significance_ratio, alpha)


def KL(t, x):
    """
    compute the KL divergence
    :param t:
    :param x:
    :return:
    """
    x = x * 1.0
    if 0 < x < t <= 1:
        if t >= 1:
            return t * np.log(t / x)
        else:
            return t * np.log(t / x) + (1 - t) * np.log((1 - t) / (1 - x))
    elif 0 <= t <= x <= 1:
        return 0
    else:
        print("t{}, x{}".format(t, x))
        raise Exception("KL Distance Error")
