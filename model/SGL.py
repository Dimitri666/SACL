import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwisev2
from base.torch_interface import TorchGraphInterface  # torch.sparse.FloatTensor
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from util.cpt import save_checkpoint, restore_best_checkpoint, restore_checkpoint
from time import strftime, localtime, time
from torch.utils.tensorboard import SummaryWriter 

# Paper: self-supervised graph learning for recommendation. SIGIR'21

class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        print(emb_size)
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()  
        print('SGL finished')

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type == 0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        # merge
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        # return user_cl_loss + item_cl_loss
        return InfoNCE(view1, view2, self.temp)

    def cal_cl_lossv2(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        #view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        #view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss
        #return InfoNCE(view1, view2, self.temp)
    
    def cal_cl_lossv3(self, idx, perturbed_mat1, perturbed_mat2):
        """SGL-torch ssl loss"""
        u_idx = torch.Tensor(idx[0]).type(torch.long).cuda()
        i_idx = torch.Tensor(idx[1]).type(torch.long).cuda()
        #print(len(u_idx), len(i_idx))
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        #view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        #view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
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
    
    
    def cal_cl_lossv4(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.Tensor(idx[0]).type(torch.long)
        i_idx = torch.Tensor(idx[1]).type(torch.long)
        
        #print(len(u_idx), len(i_idx))
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)

        user_view_1, item_view_1 = F.normalize(user_view_1, dim=1), F.normalize(item_view_1, dim=1)
        user_view_2, item_view_2 = F.normalize(user_view_2, dim=1), F.normalize(item_view_2, dim=1)
        # regard neighbors of user in the other view as positive, not neighbors in the other view as negtive
        
        user_mask = self.data.interaction_mat[u_idx,:].toarray()
        user_mask_tensor = torch.Tensor(user_mask).cuda()
        user_sim = torch.matmul(user_view_1[u_idx], torch.transpose(item_view_2,0,1))
        pos_user_sim = user_sim * user_mask_tensor
        u_cl_loss = torch.sum(-torch.log(torch.exp(pos_user_sim/self.temp).sum(1) / torch.exp(user_sim/self.temp).sum(1)))

        item_mask = self.data.interaction_mat.T[i_idx, :].toarray()
        item_mask_tensor = torch.Tensor(item_mask).cuda()
        item_sim = torch.matmul(item_view_1[i_idx], torch.transpose(user_view_2,0,1))
        pos_item_sim = item_sim * item_mask_tensor
        i_cl_loss = torch.sum(-torch.log(torch.exp(pos_item_sim/self.temp).sum(1) / torch.exp(item_sim/self.temp).sum(1)))

        return u_cl_loss + i_cl_loss 

class SGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SGL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SGL'])
        self.cl_rate = float(args['-lambda'])
        aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        #self.tb_writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name))
        print(dataset_name)
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type)
        self.device = 'cuda:0'
        self.save_period = 5
        current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime(time()))
        self.writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name + '/' + current_time))
        self.save_dir = './save/' + self.model_name + '/' + dataset_name + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        model, start_epoch = restore_checkpoint(model, self.save_dir, self.device)
        for epoch in range(start_epoch, self.maxEpoch):
            model.train()
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwisev2(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch   # [batch_size]
                #print(len(user_idx), len(pos_idx), len(neg_idx))
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * (model.cal_cl_lossv2([user_idx, pos_idx], dropped_adj1, dropped_adj2))
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'bpr_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                self.fast_evaluation(epoch)
                save_checkpoint(model,epoch,self.save_dir,self.save_period)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def final_eval(self):
        model = self.model.cuda()
        epoch = 40 # yelp 15; iFashion 15; amazon 40
        ep = (epoch//self.save_period)*self.save_period
        model = restore_best_checkpoint(ep, model, self.save_dir, self.device)
        with torch.no_grad():
            self.group_evaluation(model)