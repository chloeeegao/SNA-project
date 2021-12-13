#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm
from collections import Counter


# In[61]:


def generate_data(G, G_found):
    V, E_c, p = G_found['V'], G_found['E_c'], G_found['p']
    E = G.edges()
    E = [(i[0], i[1]) for i in E]
    degree = G.degree()
    centrality = nx.degree_centrality(G)
    cluster_coef = nx.clustering(G)
    degree_counts = Counter([G.degree(n) for n in G.nodes()])

    degree_sum_list = []
    degree_diff_list = []
    cen_sum_list = []
    if_connected = []
    freq_list = []
    sum_cluster_coef = []
    probability = []

    # size = len(E_c)
    # sample_index = random.sample(range(len(E_c)),66000)
    for i in tqdm(range(len(E_c))):
        (u, v) = E_c[i]
        if (u, v) in E:
            if_connected.append(1)
        else:
            if_connected.append(0)
        # u = str(u)
        # v = str(v)
        sum_of_degree = degree[u] + degree[v]
        sum_of_cen = centrality[u] + centrality[v]
        sum_of_freq = degree_counts[G.degree(u)] + degree_counts[G.degree(v)]
        diff_of_degree = abs(degree[u] - degree[v])
        sum_of_cluster_coef = cluster_coef[u] + cluster_coef[v]
    
        
        degree_diff_list.append(diff_of_degree)
        freq_list.append(sum_of_freq)
        degree_sum_list.append(sum_of_degree)
        cen_sum_list.append(sum_of_cen)
        sum_cluster_coef.append(sum_of_cluster_coef)

        probability.append(p[i])


    # resource allocation index 
    res_index = nx.resource_allocation_index(G, E_c)
    res_index_dict={}
    for u, v, p in res_index:
        res_index_dict[(u,v)] = p
    # jaccard coefficient
    jac_coe = nx.jaccard_coefficient(G, E_c)
    jac_coe_dict={}
    for u, v, p in jac_coe:
        jac_coe_dict[(u,v)] = p
    # adamic adar index
    adar_index_dict={}
    for m in E_c:
        try:
            for u, v, p in nx.adamic_adar_index(G, ebunch=[(m[0],m[1])]):
                adar_index_dict[(u,v)] = p
        except ZeroDivisionError:
                adar_index_dict[(m[0],m[1])] = 0
    # preferential attachment 
    pref_att = nx.preferential_attachment(G, E_c)
    pref_att_dict={}
    for u, v, p in pref_att:
        pref_att_dict[(u,v)] = p
    # # common neighbor centrality
    # com_neighbor_cen = nx.common_neighbor_centrality(G, sample)
    # com_neighbor_cen_dict={}
    # for u, v, p in com_neighbor_cen:
    #     com_neighbor_cen_dict[(u,v)] = p

    csv_dict = {'edge': E_c, 'sum of degree': degree_sum_list, 'diff of degree': degree_diff_list,
                'sum of centrality': cen_sum_list, "sum of freq": freq_list, 'sum of cluster coef': sum_cluster_coef,
                'resource allocation index':res_index_dict.values(), 'jaccard coefficient': jac_coe_dict.values(),
                'adamic adar index': adar_index_dict.values(), 'preferential attachment': pref_att_dict.values(),
                "if connected": if_connected, "label": probability}
    data_df = pd.DataFrame(csv_dict)
    data_df.to_csv('probability_regression.csv', index=False)


# In[ ]:


if __name__ == '__main__':
    file = open('k20_ep1_s1_best_sigma.pickle', 'rb')
    G_found =  pickle.load(file)
    G = nx.read_edgelist('facebook_combined.txt')
    G = nx.relabel.convert_node_labels_to_integers(G, first_label=0,ordering='default')
    G = G.to_undirected()
    # generate_train_data_for_reg(G, G_found)
    generate_data(G, G_found)
    


# In[ ]:




