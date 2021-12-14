import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import copy
import pickle
import time

from scipy.stats import entropy
from scipy.integrate import quad
from scipy.stats import norm

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# gaussian distribution
def make_gaussian(sigma, mu):
    return (lambda x: 1 / (sigma * (2 * np.pi) ** .5) *
                      np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2)))


def Y_distribution(mu, sigma, w):
    return quad(make_gaussian(sigma, mu), w - 0.5, w + 0.5)[0]


def sampled_columns(V, mu, sigma):
    col = np.zeros((len(V)))
    for i in range(len(V)):
        col[i] = Y_distribution(mu, sigma, i)
    col = col / np.sum(col)
    return col

def obfuscated(V, degree_list, E_c, p_list, k):
    print("=====calculate obfuscate=====")
    obf_number = 0
    non_obf_number = 0
    edge_matrix = np.zeros((len(V), len(V)))
    for i, (v, u) in enumerate(E_c):
        edge_matrix[v, u] = p_list[i]

    Y_dis_matrix = np.zeros((len(V), len(V)))
    print("   ======constructing Y distribution matrix=====")
    for row_number in tqdm(range(len(V))):
        row = edge_matrix[row_number, :]
        mu = np.sum(row)
        sigma = np.sqrt(np.sum(row * (1 - row)))
        if mu != 0 and sigma != 0:
            Y_dis_matrix[row_number, :] = sampled_columns(V, mu, sigma)
        else:
            Y_dis_matrix[row_number, :] = np.zeros((len(V)))

    Y_dis_matrix = Y_dis_matrix / np.sum(Y_dis_matrix, axis=0)
    Y_dis_matrix = np.nan_to_num(Y_dis_matrix)

    # calculating the number of obfuscation vertex
    print("   =====calculating the number of obfuscation vertex=====")
    for i in tqdm(range(len(V))):
        degree = degree_list[i]
        Y = Y_dis_matrix[:, degree]
        Y_entropy = entropy(Y, base=2)
        if Y_entropy >= np.log2(k):
            obf_number += 1
        else:
            non_obf_number += 1
        break
    return obf_number, non_obf_number


def train_model(dataset, original):
    feature = dataset.drop(columns=['edge', 'label'])
    prob = np.array(dataset['label'])

    scaler = StandardScaler()
    scaler.fit(feature)

    std_feature = scaler.transform(feature)

    train_x = std_feature[:10000]
    reg_train_y = prob[:10000]

    test_x = std_feature[:]
    reg_test_y = prob[:]

    # regression model
    reg = SVR(C=1.0, epsilon=0.2)
    reg.fit(train_x, reg_train_y)
    score = reg.score(test_x, reg_test_y)
    print(score)

    a = time.time()
    p_list = np.array(reg.predict(test_x))
    print(time.time()-a)

    V = np.array(original.nodes())
    E_c = np.array(dataset['edge'])
    E_c = np.array([eval(a) for a in E_c])
    # check obfuscation
    degree_list = np.array([i for (_, i) in original.degree()])
    _, non_obf_number = obfuscated(V, degree_list, E_c, p_list, k=20)
    current_eps = non_obf_number / len(V)
    print(current_eps)
    G_hat = {"V": V, "E_c": E_c, "p": p_list, "eps": current_eps}
    with open(path + f"/SNA/paper_code/result/wiki/ml_res.pickle", "wb") as fp:
        pickle.dump(G_hat, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return reg


def corss_eval(dataset1, dataset2, original):
    feature1 = dataset1.drop(columns=['edge', 'label'])
    feature2 = dataset2.drop(columns=['edge', 'label'])
    prob1 = np.array(dataset1['label'])
    prob2 = np.array(dataset2['label'])

    scaler = StandardScaler()
    scaler.fit(feature1)

    std_feature1 = scaler.transform(feature1)
    std_feature2 = scaler.transform(feature2)

    train_x = std_feature1[:20000]
    reg_train_y = prob1[:20000]

    # regression model
    reg = SVR(C=1.0, epsilon=0.2)
    reg.fit(train_x, reg_train_y)
    score = reg.score(std_feature2, prob2)
    print(score)

    p_list = np.array(reg.predict(std_feature2))
    V = np.array(original.nodes())
    E_c = np.array(dataset2['edge'])
    E_c = np.array([eval(a) for a in E_c])
    # check obfuscation
    # degree_list = np.array([i for (_, i) in original.degree()])
    # _, non_obf_number = obfuscated(V, degree_list, E_c, p_list, k=20)
    # current_eps = non_obf_number / len(V)
    # print(current_eps)
    G_hat = {"V": V, "E_c": E_c, "p": p_list, "eps": 0}
    with open(path + f"/SNA/paper_code/result/hepph/trans_ml_res.pickle", "wb") as fp:
        pickle.dump(G_hat, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    """
    # train
    condmat = pd.read_csv(path + '/SNA/data/probability_regression_wiki.csv')
    original = nx.read_edgelist(path + '/SNA/data/CA-HepPh.txt')
    original = nx.relabel.convert_node_labels_to_integers(original, first_label=0,
                                                                 ordering='default')
    reg = train_model(condmat, original)

    """
    dataset1 = pd.read_csv(path + '/SNA/data/probability_regression_CA-cond.csv')
    dataset2 = pd.read_csv(path + '/SNA/data/probability_regression_CA-hep.csv')
    original1 = nx.read_edgelist(path + '/SNA/data/CA-CondMat.txt')
    original2 = nx.read_edgelist(path + '/SNA/data/CA-HepPh.txt')
    corss_eval(dataset1, dataset2, original2)
    # corss_eval(dataset2, dataset1, original2)