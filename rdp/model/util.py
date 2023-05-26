import pandas as pd
import random
import sys
import joblib

sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score, roc_auc_score

import time
import datetime

mem = Memory("./dataset/svm_data")

@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]



# random sampling with replacement
def random_list(start, stop, length):
    if length >= 0:
        length = int(length)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))  # including start and stop
    return random_list


def aucPerformance(scores, labels, logfile=None):
    roc_auc = roc_auc_score(labels, scores)
#    print(roc_auc)
    ap = average_precision_score(labels, scores)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    if logfile:
        logfile.write("AUC-ROC: %.4f, AUC-PR: %.4f\n" % (roc_auc, ap))


    return roc_auc, ap


def tic_time():
    print("=====================================================")
    tic_datetime = datetime.datetime.now()
    print("tic_datetime:", tic_datetime)
    print("tic_datetime.strftime:", tic_datetime.strftime('%Y-%m-%d %H:%M:%S.%f'))
    tic_walltime = time.time()
    print("tic_walltime:", tic_walltime)
    tic_cpu = time.clock()
    print("tic_cpu:", tic_cpu)
    print("=====================================================\n")


    