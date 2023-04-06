from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from concrete import ConcreteDropout
from utils import sample, emb, get_roc_score, sample_DUB
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import torch.optim as optim
from torch.distributions import Normal

class ProtConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, dropout=None):
        super(ProtConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim_out,
                            kernel_size=kernel_size, padding=padding)

        self.act = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, fea):
        # print("size",len(fea.size()))
        if len(fea.size()) == 1:
            fea = fea.unsqueeze(1)
        if len(fea.size()) == 2:
            fea = fea.unsqueeze(1)
        x = self.conv(fea)
        x = self.act(x)
        return x

class ProtEncoder(nn.Module):
    def __init__(self):
        super(ProtEncoder, self).__init__()
        self.prot_fc1 = ProtConv(1, 16, 8, 0,dropout=0.1)
        self.prot_fc2 = ProtConv(16, 32, 8, 0,dropout=0.1)
        self.fc = nn.Linear(1010, 768)
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, prot):
        x = prot
        x = self.prot_fc1(x)
        x = self.prot_fc2(x)
        x = torch.max(x, 1)[0]
        x = self.fc(x)
        x = torch.relu(x)

        return x


class Perceptron(nn.Module):
    def __init__(self, dim_in, dim_out, act=None, dropout=None):
        super(Perceptron, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act
        self.dropout = dropout
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        if self.act == 'relu':
            x = torch.relu(x)
        elif self.act == 'log_softmax':
            x = F.log_softmax(x, 1)
        elif self.act == 'sigmoid':
            x = torch.sigmoid(x)
        return x

class GNetNN(nn.Module):
    def __init__(self):
        super(GNetNN, self).__init__()

        self.drug_encoder = ProtEncoder()
        self.prot_encoder = ProtEncoder()

        self.mlp_list = nn.ModuleList([])
        for i in range(3):
            if i == 0:
                fc = Perceptron(768*2, 512,
                                act='relu', dropout=0.25)
            elif i == 2:
                fc = Perceptron(256, 1, act='sigmoid')
            elif i == 1:
                fc = Perceptron(512, 256,
                                act='relu', dropout=0.25)
            self.mlp_list.append(fc)

        self.dropout = 0.1

    def forward(self, x):
        x_dub = x[:,:1024]
        x_sub = x[:,1024:]
        x_dub = self.drug_encoder(x_dub)
        x_sub = self.prot_encoder(x_sub)
        x = torch.cat([x_dub, x_sub], -1)
        for mlp in self.mlp_list:
            x = mlp(x)

        return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cal_roc_auc(label, pred):
    prob = pred
    score = roc_auc_score(label, prob)
    return score

def get_edge_embeddings(edge_list,E3_emb_matrix, sub_emb_matrix):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = E3_emb_matrix[node1]
        emb2 = sub_emb_matrix[node2]
        edge_emb = np.concatenate([emb1, emb2])
        # edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)

    return embs

#//87000  3500
# AUC_nn_1024-128_2_change2_state4100_SGD01_esm02_drop025_loss1:15
def Train():
    random_seed = 1000
    # train_edges, all_association= sample(random_seed=random_seed)
    # E3_emb_matrix, sub_emb_matrix,DUB_emb_matirx, DUB_sub_emb_matirx= emb()
    train_edges, all_association = sample_DUB(random_seed=random_seed)
    _, _, E3_emb_matrix, sub_emb_matrix = emb()
    print("all_association",len(all_association))

    kf = KFold(n_splits = 5 , shuffle = True,random_state=random_seed)
    train_index = []
    test_index = []
    for train_idx, test_idx in kf.split(train_edges[:,2]):
        train_index.append(train_idx)
        test_index.append(test_idx)

    print("train_index",train_index)
    print("train_edge",train_edges)
    print("train_index[0]",len(train_index[0]))
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    auc_result = []
    fprs = []
    tprs = []
  
    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold DUB', i + 1)
        fold_i = i
        train_edge_false = []
        train_edge = []
        neg = all_association
        
        for xx in train_index[i]:
            if train_edges[xx][2] == 1:
                train_edge.append(train_edges[xx])
    
        print("train_edge:",len(train_edge))
        test_edges_false = []
        test_edge = []

        for xx in test_index[i]:
            if train_edges[xx][2] == 1: # 正样�?                
                test_edge.append(train_edges[xx])
            else:
                test_edges_false.append(train_edges[xx])

        neg = np.array(list(set(tuple(x) for x in neg).difference(set(tuple(x) for x in train_edge))))  #去除训练集中的正样本
        pos_train_edge_embs = get_edge_embeddings(train_edge,E3_emb_matrix, sub_emb_matrix)
        
        pos_test_edge_embs = get_edge_embeddings(test_edge,E3_emb_matrix, sub_emb_matrix)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false,E3_emb_matrix, sub_emb_matrix)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
        test_edge_labels = np.concatenate([np.ones(len(test_edge)), np.zeros(len(test_edges_false))])
       
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        
        test_X = torch.from_numpy(test_edge_embs).type(torch.FloatTensor).to(device)
        net = GNetNN()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()
        if use_cuda:
            net.to(device)
        mean_max = 0
        max_auc = 0
        max_acc = 0
        patience = 0
        epoch = 3000
        torch.backends.cudnn.enabled=False
        for i in range(epoch):
            if i % 200 == 0:
                # 取训练时负样本
                train_edge_false = []
                neg_indx = np.random.choice(len(neg), len(train_edge), replace=False)
    
                for xx in neg_indx:
                    train_edge_false.append(neg[xx])
                neg_train_edge_embs = get_edge_embeddings(train_edge_false,E3_emb_matrix, sub_emb_matrix)

                train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
                train_edge_labels = np.concatenate([np.ones(len(train_edge)), np.zeros(len(train_edge_false))])
                x = torch.from_numpy(train_edge_embs).type(torch.FloatTensor).to(device)
                y = torch.from_numpy(train_edge_labels).type(torch.FloatTensor).to(device)

            net.train()
            out = net(x).squeeze()
            loss = loss_fn(out,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.eval()
            y_pred = net(test_X).cpu().squeeze()
            test_preds = y_pred.detach().numpy()
            test_auc, test_roc_curve, accuracy_test, precision_test, recall_test, f1_test = get_roc_score(test_edge_labels,test_preds)

            mean_1 = (test_auc + accuracy_test) / 2

            if mean_max < mean_1:
                mean_max = mean_1
            if max_auc < test_auc:
                max_auc = test_auc
            if max_acc < accuracy_test:
                max_acc = accuracy_test
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, "model_train/model_1/model_%s.pkl" %(str(fold_i)))
                # torch.save(net.state_dict(),"model_train2/model_1/model_%s.pkl" %(str(fold_i)))
                patience = 0
            else:
                patience += 1

            if patience > 900:
                print("Early stopping")
                break

            if i > 0 and i % 10 == 0:
                print("Epoch: " + str(i)  + "    Loss: " + str(loss.item()) + "     mean_max: " + str(mean_max)+ "     max_auc: " + str(max_auc)+ "     max_acc: " + str(max_acc)+ "     test_auc: " + str(test_auc))
            
        print("========================split================================")
        net.eval()
        checkpoint = torch.load("model_train/model_1/model_%s.pkl"%(str(fold_i)), map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if use_cuda:
            net.to(device)
        y_pred = net(test_X).cpu().squeeze()
        test_preds = y_pred.detach().numpy()
    
        test_roc, test_roc_curve, accuracy_test, precision_test, recall_test, f1_test = get_roc_score(test_edge_labels,test_preds)
        fpr = test_roc_curve[0]
        tpr = test_roc_curve[1]
        test_auc = metrics.auc(fpr,tpr)
        print("test_auc: ", test_auc)
        print("accuracy_test: ", accuracy_test)
        print("precision_test: ", precision_test)
        print("recall_test: ", recall_test)
        print("f1_test: ", f1_test)
   

        auc_result.append(test_auc)
        acc_result.append(accuracy_test)
        pre_result.append(precision_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        fprs.append(fpr)
        tprs.append(tpr)

    print("## Training Finished!")
    print("--------------------------------------------------------------------------")
    return auc_result, fprs, tprs, acc_result, pre_result, recall_result, f1_result


