# -*- coding:utf-8 -*-
import time
import tensorflow as tf
from utils import *
from sklearn.model_selection import RepeatedKFold
import numpy as np
from ClassifierOutput import *
import pandas as pd
from imblearn.combine import SMOTETomek

# Set random seed
seed = 123
np.random.seed(seed)
# tf.set_random_seed(seed)
tf.random.set_seed(seed)

def run_evaluation(X_train, y_train, X_test, y_test):
    t = time.time()
    # If the proportion of positive samples in the training set exceeds 40%, it will be balanced
    if (label_sum(y_train) > (int(len(y_train) * 0.4))):
        print("The training data does not need balance.")
        X_resampled, y_resampled = X_train, y_train
    else:
        # data sample
        X_resampled, y_resampled = SMOTETomek().fit_resample(X_train, y_train)
        # shuffle the data and labels
        state = np.random.get_state()
        np.random.shuffle(X_resampled)
        np.random.set_state(state)
        np.random.shuffle(y_resampled)

    # training classifier
    _, _, precision, recall, fmeasure, _, _ = \
        classifier_output('MLP', X_resampled, y_resampled, X_test, y_test,
                          grid_sear=True)  # False is only for debugging.

    print("precision=", "{:.5f}".format(precision),
          "recall=", "{:.5f}".format(recall),
          "f-measure=", "{:.5f}".format(fmeasure),
          "time=", "{:.5f}".format(time.time() - t))
    return precision, recall, fmeasure

def load_within_train_test(baseURL, project, mode):
    F1_list = []
    precision_list = []
    recall_list = []

    if mode == 'origin':
        # Traditional Static Code Metric(TCM)：20 dimension
        file = pd.read_csv(baseURL + project + "/Process-Binary.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif (mode == "metric"):
        # Complex Network Metric(CNM)：17 dimension
        file = pd.read_csv(baseURL + project + "/Process-Metric.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif (mode == "vector"):
        # Network Embedding Metric(NEM)：32 dimension
        file = pd.read_csv(baseURL + project+ "/Process-Vector.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif (mode == "origin_metric"):
        # TCM + CNM (TCNM): 37 dimension
        file = pd.read_csv(baseURL + project + "/Process-Binary-Metric.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif (mode == "origin_vector"):
        # TCM + NEM (TCNEM): 52 dimension
        file = pd.read_csv(baseURL + project + "/Process-Binary-Vector.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif (mode == "metric_vector"):
        # CNM + NEM (CNEM): 49 dimension
        file = pd.read_csv(baseURL + project + "/Process-Metric-Vector.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif (mode == "origin_metric_vector"):
        # TCM + CNM + NEM (TCN): 69 dimension
        file = pd.read_csv(baseURL + project + "/Process-Binary-Metric-Vector.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])
    elif mode == 'gcn':
        # GCN emb：32 dimension
        file = pd.read_csv(baseURL + project + "/gcn_emb_32.emd", sep=" ", header=None, index_col=False)
        X = np.array(file.iloc[:, 1:-1])

    origin_train_data = pd.read_csv(baseURL + project + "/Process-Binary.csv", header=0,
                                    index_col=False)
    y = np.array(origin_train_data['bug'])
    exp_cursor = 1
    kf = RepeatedKFold(n_splits=5,n_repeats=5)  # We can modify n_repeats when debugging.
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        precision, recall, fmeasure = run_evaluation(X_train, y_train, X_test, y_test)
        F1_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)

        exp_cursor = exp_cursor + 1

    avg = []
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(F1_list))

    name = ['precision', 'recall', 'F1']
    results = []
    results.append(precision_list)
    results.append(recall_list)
    results.append(F1_list)
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)
    df.to_csv('./results/'+project+'/'+mode+'.csv')

# loop eight projects
baseURL = "./data/"
projects = ['Ant','Camel','Ivy','jEdit','Lucene','Poi','Velocity','Xalan']
for i in range(len(projects)):
    print(projects[i] + " Start!")
    # mode: origin, metric, vector, origin_metric, origin_vector, metric_vector, origin_metric_vector, gcn
    load_within_train_test(baseURL, projects[i], 'gcn')
