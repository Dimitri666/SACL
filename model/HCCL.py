import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwisev2
from base.torch_interface import TorchGraphInterface  # torch.sparse.FloatTensor
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, pairwise_marginal_loss, calcRegLoss
from data.augmentor import GraphAugmentor
import pdb

# Hypergraph Contrastive Collaborative Filtering SIGIR 2022

class HCCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(HCCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['HCCL'])
        self.cl_rate = float(args['-lambda'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        hyperedge_num = int(args['-hyperedge_num']) 
        hypermap_layer = int(args['-hypermap_layer'])
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        #self.tb_writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name))
        self.model = HCCL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, hyperedge_num, hypermap_layer)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwisev2(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch   # [batch_size]
                #print(len(user_idx), len(pos_idx), len(neg_idx))
                final_user_emb, final_item_emb, local_user_emb_list, local_item_emb_list, global_user_emb_list, global_item_emb_list = model()
                user_emb, pos_item_emb, neg_item_emb = final_user_emb[user_idx], final_item_emb[pos_idx], final_item_emb[neg_idx]
                rec_loss = pairwise_marginal_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * (model.cal_cl_lossv1(local_user_emb_list, local_item_emb_list, global_user_emb_list, global_item_emb_list, [user_idx,pos_idx]))
                batch_loss = rec_loss + self.reg * calcRegLoss(model) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'bpr_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _, _, _ = self.model()
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _, _, _, _, = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class HCCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, hyperedge_num, hypermap_layer):
        super(HCCL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.hyperedge_num = hyperedge_num
        self.n_layers = n_layers
        self.hypermap_layer = hypermap_layer
        self.temp = temp
        self.norm_adj = data.norm_interation_mat
        self.embedding_dict, self.embedding_matrix = self._init_model()#, self.hypermapping_user_matrix, self.hypermapping_item_matrix
        self.act1 = nn.LeakyReLU(0.5)
        self.act2 = nn.LeakyReLU(0.5)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        embedding_matrix = nn.ParameterDict({
            'user_mat': nn.Parameter(initializer(torch.empty(self.emb_size, self.hyperedge_num))),
            'item_mat': nn.Parameter(initializer(torch.empty(self.emb_size, self.hyperedge_num))),
        })
        
        #hypermapping_user_matrix = nn.ParameterList([nn.Parameter(initializer(torch.empty(self.hyperedge_num, self.hyperedge_num))) for _ in range(self.hypermap_layer)])
        #hypermapping_item_matrix = nn.ParameterList([nn.Parameter(initializer(torch.empty(self.hyperedge_num, self.hyperedge_num))) for _ in range(self.hypermap_layer)])
        
        return embedding_dict, embedding_matrix#, hypermapping_user_matrix, hypermapping_item_matrix
    
    def forward(self):
        #pdb.set_trace()
        user_emb, item_emb = self.embedding_dict['user_emb'], self.embedding_dict['item_emb']
        user_mat, item_mat = self.embedding_matrix['user_mat'], self.embedding_matrix['item_mat']
        #hypermap_user_mat, hyper_item_mat = self.hypermapping_user_matrix, self.hypermapping_item_matrix

        user_emb_list, item_emb_list = [user_emb], [item_emb]
        local_user_emb_list, local_item_emb_list = [], []
        global_user_emb_list, global_item_emb_list = [], []

        hyper_mat_user = torch.matmul(user_emb, user_mat)
        hyper_mat_item = torch.matmul(item_emb, item_mat)
        for l in range(self.n_layers):
            local_user_emb, local_item_emb = self.local_aggregation(user_emb_list[-1], item_emb_list[-1])
            # 问题：论文中的hypergraph dependency matrix H 随着层数不同发生变化吗？在源码中是不变的，这里会随着每层的embedding不同发生变化
            #global_user_emb, global_item_emb = self.global_aggregation(user_emb_list[-1], item_emb_list[-1], user_mat, item_mat, hypermap_user_mat, hyper_item_mat)
            global_user_emb, global_item_emb = self.global_aggregationv1(user_emb_list[-1], item_emb_list[-1],  F.dropout(hyper_mat_user, p=self.drop_rate), F.dropout(hyper_mat_item, p=self.drop_rate))
            user_emb_list.append(local_user_emb+global_user_emb)
            item_emb_list.append(local_item_emb+global_item_emb)
            local_user_emb_list.append(local_user_emb)
            local_item_emb_list.append(local_item_emb)
            global_user_emb_list.append(global_user_emb)
            global_item_emb_list.append(global_item_emb)
        
        final_user_emb = sum(user_emb_list)
        final_item_emb = sum(item_emb_list)
        return final_user_emb, final_item_emb, local_user_emb_list, local_item_emb_list, global_user_emb_list, global_item_emb_list
    
    def cal_cl_loss(self, local_user_emb_list, local_item_emb_list, global_user_emb_list, global_item_emb_list, idx):
        u_idx = torch.Tensor(idx[0]).type(torch.long)
        i_idx = torch.Tensor(idx[1]).type(torch.long)
        u_loss, i_loss = 0, 0
        for l in range(len(local_user_emb_list)):
            u_loss += InfoNCE(local_user_emb_list[l][u_idx], global_user_emb_list[l][u_idx], self.temp)
            i_loss += InfoNCE(local_item_emb_list[l][i_idx], global_item_emb_list[l][i_idx], self.temp)
        return u_loss+i_loss
    
    def cal_cl_lossv1(self, local_user_emb_list, local_item_emb_list, global_user_emb_list, global_item_emb_list, idx):
        u_idx = torch.Tensor(idx[0]).type(torch.long)
        i_idx = torch.Tensor(idx[1]).type(torch.long)
        u_loss = InfoNCE(local_user_emb_list[-1][u_idx], global_user_emb_list[-1][u_idx], self.temp)
        i_loss = InfoNCE(local_item_emb_list[-1][i_idx], global_item_emb_list[-1][i_idx], self.temp)
        return u_loss+i_loss
    
    def local_aggregation(self, user_emb, item_emb):
        dropped_mat = GraphAugmentor.edge_dropout(self.norm_adj, self.drop_rate)
        dropped_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()
        local_agg_user_emb = self.act1(torch.sparse.mm(dropped_mat, item_emb))
        local_agg_item_emb = self.act1(torch.sparse.mm(torch.transpose(dropped_mat,0,1), user_emb))
        return local_agg_user_emb, local_agg_item_emb
    
    def global_aggregation(self, user_emb, item_emb, user_mat, item_mat, hypermap_user_mat, hypermap_item_mat):
        hyper_mat_user = F.dropout(torch.matmul(user_emb, user_mat), p=self.drop_rate)
        hyper_mat_item = F.dropout(torch.matmul(item_emb, item_mat), p=self.drop_rate)
   
        hyperedge_user_emb = torch.matmul(hyper_mat_user.T, user_emb)
        hyperedge_item_emb = torch.matmul(hyper_mat_item.T, item_emb)
        for l in range(self.hypermap_layer):
            hyperedge_user_emb = self.act(torch.matmul(hypermap_user_mat[l], hyperedge_user_emb)) + hyperedge_user_emb
            hyperedge_item_emb = self.act(torch.matmul(hypermap_item_mat[l], hyperedge_item_emb)) + hyperedge_item_emb
        
        hyperedge_user_emb = self.act2(torch.matmul(hyper_mat_user, hyperedge_user_emb))
        hyperedge_item_emb = self.act2(torch.matmul(hyper_mat_item, hyperedge_item_emb))
        return hyperedge_user_emb, hyperedge_item_emb
    
    def global_aggregationv1(self, user_emb, item_emb, hyper_mat_user, hyper_mat_item):
        hyperedge_user_emb = torch.matmul(hyper_mat_user.T, user_emb)
        hyperedge_item_emb = torch.matmul(hyper_mat_item.T, item_emb)
        hyperedge_user_emb = torch.matmul(hyper_mat_user, hyperedge_user_emb)
        hyperedge_item_emb = torch.matmul(hyper_mat_item, hyperedge_item_emb)
        return hyperedge_user_emb, hyperedge_item_emb