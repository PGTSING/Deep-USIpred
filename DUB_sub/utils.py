import os
from re import sub
import pandas as pd
import scipy.sparse as sp
import numpy as np
from scipy.sparse.construct import rand
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn import metrics

def returnID():
    f_2 = open(os.path.join("new_data/id_DUB.txt"), "r")
    f_3 = open(os.path.join("new_data/id_DUB_sub.txt"), "r")
    f_1 = open(os.path.join("data/id_E3_YEAST.txt"), "r")
    f_4 = open(os.path.join("data/id_sub_YEAST.txt"), "r")
    f_5 = open(os.path.join("data/id_sub_YEAST.txt"), "r")
    id_DUB = {}
    id_DUB_sub = {}
    id_sub = {}
    id_e3 = {}
    dub_name = {}
    dub_name2id = {}
    sub_name2id = {}
    for x in f_1:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = x[1].strip('\n')
        id_e3[x2] = x1

    for x in f_2:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = x[1].strip('\n')
        id_DUB[x2] = x1
        dub_name2id[x1] = x2

    for x in f_3:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = x[1].strip('\n')
        id_DUB_sub[x2] = x1
        sub_name2id[x1] = x2

    for x in f_4:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = x[1].strip('\n')
        id_sub[x2] = x1

    return id_DUB,id_DUB_sub, id_sub, id_e3,dub_name,dub_name2id,sub_name2id

def sample_DUB( random_seed):
    f_4 = open(os.path.join("new_data/id_DUB_sub_link.txt"), "r")
    id_e3 ,id_sub,_, _,_ ,dub_name2id,sub_name2id= returnID()
    matrix = [([0] * len(id_sub)) for i in range(len(id_e3))]
    known_associations = []
    unknown_associations = []
    all_associations = []
    for x in f_4:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n'))
        matrix[x1][x2] = 1
    f_4.close()

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                known_associations.append([i, j])
            else:
                unknown_associations.append([i, j])
    matrix = sp.csr_matrix(matrix)

    npd1 = np.array(unknown_associations)
    npd2 = np.array(known_associations)
    # npd3 = np.array(all_associations)
    df1 = pd.DataFrame(npd1,columns=['dub','sub'])
    df1['label'] = 0
    df2 = pd.DataFrame(npd2,columns=['dub','sub'])
    df2['label'] = 1
    random_negative = df1.sample(n=df2.shape[0], random_state=random_seed, axis=0)
    sample_df = df2.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    all_associations = df2.append(df1)
    # all_associations = df1
    all_associations.reset_index(drop=True, inplace=True)

    print(sample_df)
    print(all_associations.values)
    return sample_df.values, all_associations.values


def sample( random_seed):
    f_4 = open(os.path.join("data/id_E3_sub_YEAST.txt"), "r")
    # f_4 = open(os.path.join("data2/id_E3_sub_HUMAN_remove_1.txt"), "r")

    id_DUB ,id_DUB_sub,id_sub, id_e3,_,dub_name2id,sub_name2id = returnID()
    matrix = [([0] * len(id_sub)) for i in range(len(id_e3))]
    known_associations = []
    unknown_associations = []
    all_associations = []
    for x in f_4:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n'))
        matrix[x1][x2] = 1
    f_4.close()

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                known_associations.append([i, j])
                # known_associations.append([id_e3.get(i), id_sub.get(j)])
            else:
                unknown_associations.append([i, j])
                # unknown_associations.append([id_e3.get(i), id_sub.get(j)])
    matrix = sp.csr_matrix(matrix)
    npd1 = np.array(unknown_associations)
    npd2 = np.array(known_associations)
    # npd3 = np.array(all_associations)
    df1 = pd.DataFrame(npd1,columns=['E3','sub'])
    df1['label'] = 0
    df2 = pd.DataFrame(npd2,columns=['E3','sub'])
    df2['label'] = 1
    random_negative = df1.sample(n=df2.shape[0], random_state=random_seed, axis=0)
    sample_df = df2.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    all_associations = df2.append(df1)
    # all_associations = df1
    all_associations.reset_index(drop=True, inplace=True)
    sample_df.to_csv("test_pairs.csv",index=False,header=None,sep="\t")
    print(sample_df)
    print(all_associations.values)
    return sample_df.values, all_associations.values

def emb():
    id_DUB ,id_DUB_sub, id_sub, id_e3, _, dub_name2id,sub_name2id = returnID()
    file1 = open("../../data_extract/E3_TAPE_features_YEAST_01.txt")
    E3 = {}
  
    for line in file1:
        embed = line.strip().split(' ')
        E3[embed[0]] = []
        for i in range(1, len(embed), 1):
            E3[embed[0]].append(float(embed[i]))
    # print(E3)
    E3_emb_list = []
    for node in id_e3.keys():
        node_emb = E3[node]
        E3_emb_list.append(node_emb)
    E3_emb_matirx = np.vstack(E3_emb_list)
   #===================================================================

    file3 = open("../../data_extract/DUB_seqVec_features_01.txt")
    # file3 = open("DUB_features1.txt")
    DUB = {}
    # file1.readline()
    for line in file3:
        embed = line.strip().split(' ')
        DUB[embed[0]] = []
        for i in range(1, len(embed), 1):
            DUB[embed[0]].append(float(embed[i]))
    # print(E3)
    DUB_emb_list = []
    for node in id_DUB.keys():
        node_emb = DUB[node]
        DUB_emb_list.append(node_emb)
    DUB_emb_matirx = np.vstack(DUB_emb_list)

   #===================================================================
    DUB_sub = {}
    file123 = open("../../data_extract/DUB_sub_seqVec_features_01.txt")
    # file123 = open("DUB_sub_features1.txt")
    id = 0
    for line in file123:
        embed = line.strip().split(' ')
        DUB_sub[embed[0]] = []
        for i in range(1, len(embed), 1):
            DUB_sub[embed[0]].append(float(embed[i]))
    # print(E3)
    DUB_sub_emb_list = []
    for node in id_DUB_sub.keys():
        node_emb = DUB_sub[node]
        DUB_sub_emb_list.append(node_emb)
    DUB_sub_emb_matirx = np.vstack(DUB_sub_emb_list)
   #===================================================================

    sub_embed_dict = {}
    file12 = open("../../data_extract/E3_sub_TAPE_features_YEAST_01.txt")
    # file12 = open("sub_features.txt")
    
    for line in file12:
        embed = line.strip().split(' ')
        sub_embed_dict[embed[0]] = []
        for i in range(1, len(embed), 1):
            sub_embed_dict[embed[0]].append(float(embed[i]))
    # print(E3)
    sub_emb_list = []
    for node in id_sub.keys():
        node_emb = sub_embed_dict[node]
        sub_emb_list.append(node_emb)
    sub_emb_matirx = np.vstack(sub_emb_list)
    return E3_emb_matirx, sub_emb_matirx, DUB_emb_matirx, DUB_sub_emb_matirx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score(test_edge_labels,test_preds):
    results_test = [0 if j < 0.5 else 1 for j in np.squeeze(test_preds)]
  
    accuracy_test = metrics.accuracy_score(test_edge_labels, results_test)
    precision_test = metrics.precision_score(test_edge_labels, results_test)
    recall_test = metrics.recall_score(test_edge_labels, results_test)
    f1_test = metrics.f1_score(test_edge_labels, results_test)

    roc_score = roc_auc_score(test_edge_labels,test_preds)
    # ap_score = average_precision_score(labels_all,preds_all)
    roc_curve_tuple = roc_curve(test_edge_labels,test_preds)
    # plot_roc_curve(labels_all,preds_all)


    return roc_score,roc_curve_tuple, accuracy_test, precision_test, recall_test, f1_test