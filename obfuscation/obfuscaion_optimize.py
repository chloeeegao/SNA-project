import os
import networkx as nx
import numpy as np
import copy
import pickle
import time

from tqdm import tqdm
from scipy.stats import entropy
from scipy.integrate import quad
from scipy.stats import norm

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Uniqueness
"""


# define the distance
def distance(a, b):
    return np.abs(a - b)


# define the gaussian distribution
def gaussian_dis(mu, sigma, x):
    return 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# define the uniqueness score for each property value in G
# rewrite the code for distance calculation
def uniqueness_opt(V, degree_list, sigma, mu):
    """
    :param V:
    :param degree_lsit:
    :param sigma:
    :param mu:
    :return:
    """
    uniqueness_list = []
    degree_array = np.array(degree_list)
    print("=====calculating the uniqueness=====")
    for i in tqdm(range(len(V))):
        degree_v = degree_list[i]
        commonness_v = np.sum(gaussian_dis(mu, sigma, degree_array - degree_v))
        uniqueness_list.append(1 / commonness_v)
    return uniqueness_list


# get H (vertices with largest uniqueness)
def H_collector(V, uni_list, eps):
    """
    input: vertice list, uniqueness list, eps
    output: H
    """
    V_with_uni = dict(zip(V, uni_list))
    sorted_V = sorted(V_with_uni.items(), key=lambda item: item[1], reverse=True)
    H = [i for (i, _) in sorted_V]
    number = int(np.ceil(eps * len(V) / 2))
    H = H[:number]
    return H


# normalization of uniqueness list
def Q_calculate(uni_list):
    uni_list = np.array(uni_list)
    Q_list = uni_list / np.sum(uni_list)
    return Q_list


# find nearest value in Q
def find_nearest(Q, value):
    diff = np.abs(Q - value)
    index = np.where(diff == min(diff))
    # ind = np.random.choice(index[0])
    return index[0]


# sample from Q
def sample_from_Q(mu, sigma, Q):
    high = gaussian_dis(mu, sigma, x=mu)
    while True:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, high)
        # noise = np.random.normal(mu, sigma, 1)
        if y <= gaussian_dis(mu, sigma, x):
            return find_nearest(Q, x)


"""
Search
"""


# compute u v
def sigma_cal(E_c, sigma, uniqueness_list):
    U_E = []
    for e in E_c:
        (u, v) = e
        U = (uniqueness_list[u] + uniqueness_list[v]) / 2
        U_E.append(U)

    sigma_e_list = list(sigma * len(E_c) * (np.array(U_E) / np.sum(np.array(U_E))))
    return sigma_e_list


# gaussian distribution
def make_gaussian(sigma, mu):
    return (lambda x: 1 / (sigma * (2 * np.pi) ** .5) *
                      np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2)))


# R distribution
def R_distribution(mu, sigma, r):
    if (r >= 0) | (r <= 1):
        return gaussian_dis(mu, sigma, r) / quad(make_gaussian(sigma, mu=0), 0, 1)[0]
    else:
        return 0


def sample_from_R(mu, sigma):
    while True:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, R_distribution(mu, sigma, r=mu))
        if y < R_distribution(mu, sigma, x):
            return x


def Y_distribution(mu, sigma, w):
    return quad(make_gaussian(sigma, mu), w - 0.5, w + 0.5)[0]


def sample_from_Y(mu, sigma, w):
    while True:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, Y_distribution(mu, sigma, w=w))
        if y < Y_distribution(mu, sigma, x):
            return x


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
            print("point %d not obfuscated with degree %d:" % (i, degree))

    return obf_number, non_obf_number


def random_edge_list_generate(list1, list2, E, E_c):
    for i in range(len(list1)//3):
        item1 = list1[i]
        for j in range(len(list2)//3):
            item2 = list2[j]
            if item1 != item2:
                edge = (min(item1, item2), max(item1, item2))
                if edge not in E:
                    E_c.add(edge)
                else:
                    try:
                        E_c.remove(edge)
                    except Exception:
                        pass

    return E, E_c


# main function
# main function for algorithm 2
def generate_obfuscation(V, E, degree_list, c, mu, sigma, eps, k, q, epoch):
    uniqueness_list = uniqueness_opt(V, degree_list, sigma, mu)
    # H = H_collector(V, uniqueness_list, eps)
    Q = Q_calculate(uniqueness_list)
    mu_Q, sigma_Q = norm.fit(Q)
    mu_Q = round(mu_Q, 4)
    sigma_Q = round(sigma_Q, 4)
    E = set([(i, j) for (i, j) in E.tolist()])
    eps_hat = np.inf
    G_hat = {}
    for epoch in range(epoch):
        E_c = copy.deepcopy(E)
        while len(E_c) <= int(c * (len(E))):
            list1 = sample_from_Q(mu_Q*0.8, sigma_Q*1.1, Q)
            list2 = sample_from_Q(mu_Q*1.1, sigma_Q*1.1, Q)
            E, E_c = random_edge_list_generate(list1, list2, E, E_c)
            time.sleep(0.0000005)
            print('\r The size of E_c is growing %d' % len(E_c), end='')
        sigma_e_list = sigma_cal(E_c, sigma, uniqueness_list)
        p_list = []
        E_c = list(E_c)
        print("\n=====computing p_e for e in E_C=====")
        for i in tqdm(range(len(E_c))):
            e = E_c[i]
            sigma_e = sigma_e_list[i]
            w = np.random.uniform(0, 1)
            if w < q:
                r_e = np.random.uniform(0, 1)/2
            else:
                r_e = sample_from_R(mu, sigma_e)/2
            if e in E:
                p_e = 1 - r_e
            else:
                p_e = r_e
            p_list.append(p_e)
        G_hat = {"V": V, "E_c": E_c, "p": p_list}
        with open(path + f"/SNA/paper_code/result/wiki/k20_ep1_s{sigma}.pickle", "wb") as fp:
            pickle.dump(G_hat, fp, protocol=pickle.HIGHEST_PROTOCOL)
        _, non_obf_number = obfuscated(V, degree_list, E_c, p_list, k)
        current_eps = non_obf_number / len(V)
        i = 0
        print(current_eps)
        G_hat = {"V": V, "E_c": E_c, "p": p_list, "eps": current_eps}
        if current_eps <= eps and current_eps < eps_hat:
            i += 1
            eps_hat = current_eps
            G_hat = {"V": V, "E_c": E_c, "p": p_list, "eps": current_eps}
            print("epoch{}-->eps_hat={}".format(epoch, eps_hat))
        with open(path + f"/SNA/paper_code/result/wiki/k20_ep1_s{sigma}.pickle", "wb") as fp:
            pickle.dump(G_hat, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return eps_hat, G_hat


def k_eps_obfuscation(V, E, degree_list, k, eps, c, mu, q):
    sigma_l = 0
    sigma_u = 1
    while True:
        a = time.time()
        eps_hat, G_hat = generate_obfuscation(V, E, degree_list, c, mu, sigma_u, eps, k, q, epoch=1)
        print(time.time() - a)
        if eps_hat == np.inf:
            sigma_u = 2 * sigma_u
        else:
            break
    G_found = G_hat
    with open(path + f"/SNA/paper_code/result/wiki/k20_ep3_s{sigma_u}_best_sigma.pickle", "wb") as fp:
        pickle.dump(G_hat, fp, protocol=pickle.HIGHEST_PROTOCOL)
    timer = 0
    while (sigma_l < sigma_u) and timer <= 5:
        sigma = (sigma_l + sigma_u)
        eps_hat, G_hat = generate_obfuscation(V, E, degree_list, c, mu, sigma_u, eps, k, q, epoch=1)
        if eps_hat == np.inf:
            sigma_l = sigma
        else:
            G_found = G_hat
            sigma_u = sigma
        timer += 1

    G_found['eps'] = eps_hat
    with open(path + f"/SNA/paper_code/result/wiki/k20_ep3_s{sigma_u}_best.pickle", "wb") as fp:
        pickle.dump(G_found, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return G_found


if __name__ == '__main__':
    """
    file = open(path + '/SNA/paper_code/G_found_GR_QC.pickle', 'rb')
    G_found = pickle.load(file)
    print(G_found)
    """

    G = nx.read_edgelist(path + '/SNA/data/facebook_combined.txt')
    G = G.to_undirected()
    reindexed_graph = nx.relabel.convert_node_labels_to_integers(G, first_label=0,
                                                                 ordering='default')

    E = np.array(reindexed_graph.edges())
    V = np.array(reindexed_graph.nodes())
    D = reindexed_graph.degree()
    degree_list = np.array([i for (_, i) in D])
    G_found = k_eps_obfuscation(V, E, degree_list, k=20, eps=1e-3, c=2, mu=0, q=0.01)
