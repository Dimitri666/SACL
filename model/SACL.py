import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_sparse import SparseTensor
from base.graph_recommender import GraphRecommender
from data.augmentor import GraphAugmentor
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwisev2
from base.torch_interface import TorchGraphInterface  # torch.sparse.FloatTensor
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.GNNaugmenter import EMA, set_requires_grad, update_moving_average
from util.cpt import save_checkpoint, restore_best_checkpoint, restore_checkpoint
from util.vis import vis_2d
from time import strftime, localtime, time
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter 

class SACL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SACL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SACL'])
        self.lr_decay = float(self.config['lrDecay'])
        self.lr_decay_step = float(self.config['lrDecayStep'])
        self.cl_rate = float(args['-lambda'])
        self.beta = float(args['-beta'])
        self.alpha = float(args['-alpha'])
        self.n_layers = n_layers = int(args['-n_layer'])
        self.temp = float(args['-temp'])
        self.aug_type = int(args['-aug_type'])
        self.drop_rate = float(args['-drop_rate'])
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda() 
        self.num_interaction = int(self.sparse_norm_adj._values().size()[0])
        self.n_negs = 1
        self.device = 'cuda:0'
        self.save_period = 5
        current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime(time()))
        self.writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name + '/' + current_time))
        self.save_dir = './save/' + self.model_name + '/' + dataset_name + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.model = SACL_Encoder(self.data, self.emb_size, n_layers, self.temp, self.beta, self.maxEpoch, self.aug_type, self.drop_rate, self.alpha)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)    
        model, start_epoch = restore_checkpoint(model, self.save_dir, self.device)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay)
        for epoch in range(start_epoch,self.maxEpoch):
            model.train()
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            clloss, recloss, aloss = 0,0,0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, self.n_negs)):
                user_idx, pos_idx, neg_idx = batch 
                rec_user_emb, rec_item_emb, aug_user_emb, aug_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                #user_emb, pos_item_emb, neg_item_emb = self.negative_mixup(rec_user_emb, rec_item_emb, user_idx, pos_idx, neg_idx)

                cl_loss = model.calc_lossv3([user_idx, pos_idx], dropped_adj1, dropped_adj2) 
          
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                reg = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)

                loss = rec_loss + self.cl_rate * cl_loss + reg
                aloss += loss.item()
                clloss += cl_loss.item()
                recloss += rec_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_moving_average()

                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item(), 'rec_loss', rec_loss.item(), 'loss', loss.item())
            #scheduler.step()
            model.eval()
    
            with torch.no_grad():
                self.user_emb, self.item_emb, _,_ = self.model()
                self.fast_evaluation(epoch)
                save_checkpoint(model,epoch,self.save_dir,self.save_period)
                
            
            recall = self.performance['Recall']
            ndcg = self.performance['NDCG']
            aloss /= self.num_interaction
            recloss /= self.num_interaction
            clloss /= self.num_interaction
            self.writer.add_scalar('loss', aloss, epoch)
            self.writer.add_scalar('rec_loss', recloss, epoch)
            self.writer.add_scalar('cl_loss', clloss, epoch)
            self.writer.add_scalar('Recall', recall, epoch)
            self.writer.add_scalar('NDCG', ndcg, epoch)
            '''
            if epoch == 66: 
                ep = (epoch//self.save_period)*self.save_period
                model = restore_best_checkpoint(ep, model, self.save_dir, self.device)    
                with torch.no_grad():            
                    self.group_eval(model)
            '''
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_,_ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def final_eval(self):
        model = self.model.cuda()
        epoch = 55 # yelp 65; iFashion 20; amazon 55
        ep = (epoch//self.save_period)*self.save_period
        model = restore_best_checkpoint(ep, model, self.save_dir, self.device)
        with torch.no_grad():
            self.group_evaluation(model)
    '''
    def vis(self):
        with torch.no_grad():
            model = self.model.cuda()
            epoch = 65
            model = restore_best_checkpoint(epoch, model, self.save_dir, self.device)
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            user_emb1, item_emb1, _,_ = model(dropped_adj1)
            user_emb2, item_emb2, _,_ = model(dropped_adj2)
            data = torch.cat([user_emb1[:10,:], user_emb2[:10,:], user_emb2[10:100,:]], dim=0).cpu()
            label = np.zeros((data.shape[0],))
            for k in range(10,20):
                label[k]=1
            for k in range(20,110):
                label[k]=2
            save_path = self.save_dir + 'user_emb2d.png'
            vis_2d(data, label, 'user_embs', save_path)
            
            data_ = torch.cat([item_emb1[:10,:], item_emb2[:10,:], user_emb2[10:100,:]], dim=0).cpu()
            save_path = self.save_dir + 'item_emb2d.png'
            vis_2d(data_, label, 'item_embs', save_path)
       ''' 
class SACL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, temp, beta, maxepoch, aug_type, drop_rate, alpha):
        super(SACL_Encoder, self).__init__()
        self.temp = temp
        self.n_laysers = n_layers
        self.data = data
        self.aug_type = aug_type
        self.drop_rate = drop_rate
        self.inter_tensor = TorchGraphInterface.convert_to_sparsetensor(self.data.interaction_mat).cuda()
        self.inter_tensorT = TorchGraphInterface.convert_to_sparsetensor(self.data.interaction_mat.T).cuda()
        self.main_model = Encoder(data, emb_size, n_layers, temp)
        self.augmenter = copy.deepcopy(self.main_model)
        self.k = alpha
        set_requires_grad(self.augmenter, False)
        self.aug_updater = EMA(beta, maxepoch)
    
    def reset_moving_average(self):
        del self.augmenter
        self.augmenter = None

    def update_moving_average(self):
        assert self.augmenter is not None, 'augmenter has not been created yet'
        update_moving_average(self.aug_updater, self.augmenter, self.main_model)

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
        rec_user_emb, rec_item_emb = self.main_model(perturbed_adj)
        with torch.no_grad():
            aug_user_emb, aug_item_emb = self.augmenter(perturbed_adj)
        
        return rec_user_emb, rec_item_emb, aug_user_emb, aug_item_emb
        
    def calc_loss(self, idx, user_view_1, item_view_1, user_view_2, item_view_2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def calc_lossv2(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1, aug_uview_1, aug_iview_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2, aug_uview_2, aug_iview_2 = self.forward(perturbed_mat2)
        
        # within network
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)

        # cross network
        user_cl_loss1 = InfoNCE(user_view_1[u_idx], aug_uview_1[u_idx], self.temp)
        item_cl_loss1 = InfoNCE(item_view_1[i_idx], aug_iview_1[i_idx], self.temp)
        return user_cl_loss + item_cl_loss + user_cl_loss1 + item_cl_loss1

    def calc_lossv3(self, idx, perturbed_mat1, perturbed_mat2):
        # weighted negtive
        
        u_idx = torch.Tensor(idx[0]).type(torch.long)
        i_idx = torch.Tensor(idx[1]).type(torch.long)
        
         #print(len(u_idx), len(i_idx))
        user_view_1, item_view_1, aug_uview_1, aug_iview_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2, aug_uview_2, aug_iview_2 = self.forward(perturbed_mat2)

        user_view_1, item_view_1 = F.normalize(user_view_1, dim=1), F.normalize(item_view_1, dim=1)
        user_view_2, item_view_2 = F.normalize(user_view_2, dim=1), F.normalize(item_view_2, dim=1)
        aug_uview_1, aug_iview_1 = F.normalize(aug_uview_1, dim=1), F.normalize(aug_iview_1, dim=1)
        aug_uview_2, aug_iview_2 = F.normalize(aug_uview_2, dim=1), F.normalize(aug_iview_2, dim=1)
        
        user_view_1 = user_view_1[u_idx]
        item_view_1 = item_view_1[i_idx]
        user_view_2 = user_view_2[u_idx]
        item_view_2 = item_view_2[i_idx]
        aug_uview_1 = aug_uview_1[u_idx] 
        aug_iview_1 = aug_iview_1[i_idx]
        aug_uview_2 = aug_uview_2[u_idx] 
        aug_iview_2 = aug_iview_2[i_idx]
        
        # cross mask
        user_mask_tensor = self.inter_tensor.index_select(0,u_idx).to_dense()
        user_mask_tensor = torch.matmul(user_mask_tensor, user_mask_tensor.transpose(0,1))
        user_mask_tensor = torch.exp(-self.k*user_mask_tensor) 

        item_mask_tensor = self.inter_tensorT.index_select(0,i_idx).to_dense()
        item_mask_tensor = torch.matmul(item_mask_tensor, item_mask_tensor.transpose(0,1))
        item_mask_tensor = torch.exp(-self.k*item_mask_tensor) 
        
        # wighin network
        pos_score = (user_view_1 * user_view_2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temp)
        ttl_score = torch.matmul(user_view_1, user_view_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temp)
        ttl_score = (ttl_score * user_mask_tensor).sum(1)
        u_cl_loss = -torch.sum(torch.log(pos_score / ttl_score))

        pos_score = (item_view_1 * item_view_2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temp)
        ttl_score = torch.matmul(item_view_1, item_view_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temp).sum(dim=1)
        ttl_score = (ttl_score * item_mask_tensor).sum(1)
        i_cl_loss = -torch.sum(torch.log(pos_score / ttl_score))

        # cross network
        pos_score = (user_view_1 * aug_uview_1).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temp)
        ttl_score = torch.matmul(user_view_1, aug_uview_1.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temp)
        ttl_score = (ttl_score * user_mask_tensor).sum(1)
        u_cl_loss1 = -torch.sum(torch.log(pos_score / ttl_score))

        pos_score = (item_view_1 * aug_iview_1).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temp)
        ttl_score = torch.matmul(item_view_1, aug_iview_1.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temp)
        ttl_score = (ttl_score * item_mask_tensor).sum(1)
        i_cl_loss1 = -torch.sum(torch.log(pos_score / ttl_score))
        
        return u_cl_loss + i_cl_loss + u_cl_loss1 + i_cl_loss1


class Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, temp):
        super(Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()  

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        #item_embs = [self.embedding_dict['item_emb']]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)    
            #item_embs.append(ego_embeddings[self.data.user_num:])

        all_embeddings1 = torch.stack(all_embeddings, dim=1)
        
        all_embeddings1 = torch.mean(all_embeddings1, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings1, [self.data.user_num, self.data.item_num])
        
        return user_all_embeddings, item_all_embeddings
        
