import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwisev2
from base.torch_interface import TorchGraphInterface  # torch.sparse.FloatTensor
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from torch.utils.tensorboard import SummaryWriter 


# Paper: self-supervised graph learning for recommendation. SIGIR'21

class DiffGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DiffGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DiffGCL'])
        self.cl_rate = float(args['-lambda'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        alpha = float(args['-alpha'])
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        #self.tb_writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name))
        self.model = DiffGCL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type, alpha)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            
            for n, batch in enumerate(next_batch_pairwisev2(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch   # [batch_size]
                #print(len(user_idx), len(pos_idx), len(neg_idx))
                rec_user_emb1, rec_item_emb1, rec_user_emb2, rec_item_emb2 = model()
                user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb1[user_idx], rec_item_emb1[pos_idx], rec_item_emb1[neg_idx]
                user_emb2, pos_item_emb2, neg_item_emb2 = rec_user_emb2[user_idx], rec_item_emb2[pos_idx], rec_item_emb2[neg_idx]

                # sum of emb of origin and augment
                user_emb = user_emb1 + user_emb2
                pos_item_emb = pos_item_emb1 + pos_item_emb2
                neg_item_emb = neg_item_emb1 + neg_item_emb2

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                cl_loss = self.cl_rate * model.cal_cl_lossv2([user_idx, pos_idx], rec_user_emb1, rec_item_emb1, rec_user_emb2, rec_item_emb2)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'bpr_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb1, self.item_emb1, self.user_emb2, self.item_emb2 = self.model()
                self.fast_evaluation(epoch)
        self.user_emb1, self.item_emb1, self.user_emb2, self.item_emb2 = self.best_user_emb1, self.best_item_emb1, self.best_user_emb2, self.best_item_emb2

    def save(self):
        with torch.no_grad():
            self.best_user_emb1, self.best_item_emb1, self.best_user_emb2, self.best_item_emb2 = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        user_emb = self.user_emb1 + self.user_emb2
        item_emb = self.item_emb1 + self.item_emb2
        score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
        return score.cpu().numpy()

class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.Sigmoid()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        u_aug, s_aug, v_aug = adj[0], adj[1], adj[2]
        out = u_aug @ s_aug @ (v_aug.T @ feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class DiffGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type, alpha):
        super(DiffGCL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.alpha = alpha
        self.t = 0.4
        self.ppr = True
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        if self.ppr:
            #self.aug_adjs = self.diffusion_graph_PPR() # u, s, v
            self.diff_layer = GCNLayer(self.emb_size, self.emb_size, False)
            self.aug_adjs = self.diffusion_graph_PPRv2() # u1, s1, v1, u2, s2, v2
        else:
            self.aug_adjs = self.diffusion_graph_heat()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb_ori': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb_ori': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
            'user_emb_aug': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb_aug': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict
    
    def diffusion_graph_PPR(self):
        adj_mat = self.data.convert_to_laplacian_mat(self.data.interaction_mat)
        Transt = TorchGraphInterface.convert_sparse_mat_to_tensor(adj_mat)
        tot_nodes = Transt.shape[0]
        # PPR
        in_ind = torch.arange(tot_nodes, dtype=torch.long).unsqueeze(0).repeat(2,1)
        in_val = torch.ones(tot_nodes, dtype=torch.float32)
        in_siz = torch.Size([tot_nodes,tot_nodes])
        identity = torch.sparse_coo_tensor(in_ind, in_val, in_siz)
        In_aT = identity - (1-self.alpha)*Transt
        u,s,vt = torch.svd_lowrank(In_aT, 64)
        s_1 = torch.inverse(torch.diag(s))
        return self.alpha*vt.cuda(), s_1.cuda(), u.cuda()   

    def diffusion_graph_heat(self):
        # heat kernel
        adj_mat = self.data.ui_adj
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = adj_mat.dot(d_mat_inv)
        #norm_adj_mat = sp.csc_matrix(norm_adj_mat)
        aug_graph = self.t*norm_adj_mat # 这里没有减去t
        aug_tensor = TorchGraphInterface.convert_sparse_mat_to_tensor(aug_graph)
        aug_tensor = torch.exp(aug_tensor)
        return aug_tensor.cuda()
    
    def diffusion_graph_PPRv2(self):
        ui_mat = self.data.normalize_graph_mat(self.data.interaction_mat)
        iu_mat = self.data.normalize_graph_mat(self.data.interaction_mat.T)
        ui_Transt = TorchGraphInterface.convert_sparse_mat_to_tensor(ui_mat)
        u1, s1, v1t = torch.svd_lowrank(ui_Transt, 6)
        iu_Transt = TorchGraphInterface.convert_sparse_mat_to_tensor(iu_mat)
        u2, s2, v2t = torch.svd_lowrank(iu_Transt, 6)
        return u1.cuda(), torch.diag(s1).cuda(), v1t.cuda(), u2.cuda(), torch.diag(s2).cuda(), v2t.cuda()

    def forward(self):
        # origin graph 
        ego_embeddings_ori = torch.cat([self.embedding_dict['user_emb_ori'], self.embedding_dict['item_emb_ori']], 0)
        all_embeddings = [ego_embeddings_ori]
        for k in range(self.n_layers):
            ego_embeddings_ori = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings_ori)
            all_embeddings.append(ego_embeddings_ori)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings_ori, item_all_embeddings_ori = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        # diffusion graph
        if self.ppr:
            #ego_embeddings_aug = torch.cat([self.embedding_dict['user_emb_aug'], self.embedding_dict['item_emb_aug']], 0)
            #all_embeddings_aug = self.diff_layer(ego_embeddings_aug, self.aug_adjs)
            user_emb = self.embedding_dict['user_emb_aug']
            item_emb = self.embedding_dict['item_emb_aug']
            u1, s1, v1, u2, s2, v2 = self.aug_adjs
            #print(u1.shape, s1.shape, v1.shape, u2.shape, v2.shape, user_emb.shape, item_emb.shape)
            user_all_embeddings_aug = self.diff_layer(item_emb, self.aug_adjs[0:3])
            item_all_embeddings_aug = self.diff_layer(user_emb, self.aug_adjs[3:])
        else:
            ego_embeddings_aug = torch.cat([self.embedding_dict['user_emb_aug'], self.embedding_dict['item_emb_aug']], 0)
            all_embeddings_aug = self.aug_adjs @ ego_embeddings_aug
        
        #user_all_embeddings_aug, item_all_embeddings_aug = torch.split(all_embeddings_aug, [self.data.user_num, self.data.item_num])

        return user_all_embeddings_ori, item_all_embeddings_ori, user_all_embeddings_aug, item_all_embeddings_aug

    def cal_cl_loss(self, idx, user_view_1, item_view_1, user_view_2, item_view_2):
        """merge loss"""
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        
        # merge
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
      
        return InfoNCE(view1, view2, self.temp)

    def cal_cl_lossv2(self, idx, user_view_1, item_view_1, user_view_2, item_view_2):
        """not merge loss"""
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
       
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss
       
    
    def cal_cl_lossv3(self, idx, user_view_1, item_view_1, user_view_2, item_view_2):
        """SGL-torch ssl loss, negative samples is beyond batch samples"""
        u_idx = torch.Tensor(idx[0]).type(torch.long).cuda()
        i_idx = torch.Tensor(idx[1]).type(torch.long).cuda()
       
        user_view_1, item_view_1 = F.normalize(user_view_1, dim=1), F.normalize(item_view_1, dim=1)
        user_view_2, item_view_2 = F.normalize(user_view_2, dim=1), F.normalize(item_view_2, dim=1)

        user_emb1 = user_view_1[u_idx]
        user_emb2 = user_view_2[u_idx]
        item_emb1 = item_view_1[i_idx]
        item_emb2 = item_view_2[i_idx]

        pos_user_rating = (user_emb1*user_emb2).sum(dim=1)
        ttl_user_rating = torch.matmul(user_emb1, torch.transpose(user_view_2,0,1))
        ssl_logits_user = ttl_user_rating - pos_user_rating[:,None]

        pos_item_rating = (item_emb1*item_emb2).sum(dim=1)
        ttl_item_rating = torch.matmul(item_emb1, torch.transpose(item_view_2,0,1))
        ssl_logits_item = ttl_item_rating - pos_item_rating[:,None]

        clogits_user = torch.logsumexp(ssl_logits_user / self.temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.temp, dim=1)
        infonce_loss = torch.sum(clogits_user + clogits_item)

        return infonce_loss