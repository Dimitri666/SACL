import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwisev2
from base.torch_interface import TorchGraphInterface  # torch.sparse.FloatTensor
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from time import strftime, localtime, time
from torch.utils.tensorboard import SummaryWriter 

# Paper: self-supervised graph learning for recommendation. SIGIR'21

class ADGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ADGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['ADGCL'])
        self.cl_rate = float(args['-lambda'])
        #aug_type = self.aug_type = int(args['-augtype'])
        b_temp = float(args['-b_temp'])
        drop_rate = float(args['-droprate'])
        self.reg_drop = float(args['-reg_drop'])
        #self.mlp_layer = float(args['-mlp_layer'])
        n_layers = int(args['-n_layer'])
        self.temp = float(args['-temp'])
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda() 
        self.num_interaction = int(self.sparse_norm_adj._values().size()[0])
        current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime(time()))
        self.writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name + '/' + current_time))
        self.model = ADGCL_Encoder(self.data, self.emb_size, drop_rate, n_layers, self.temp)
        self.ed_augmenter = BernEdgeAugmenter(self.emb_size, b_temp)
        self.nd_augmenter = BernNodeAugmenter(self.emb_size, b_temp)

    def train(self):
        model = self.model.cuda()
        ed_augmenter = self.ed_augmenter.cuda()
        nd_augmenter = self.nd_augmenter.cuda()
        model_optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        ed_augmenter_optimizer = torch.optim.Adam(ed_augmenter.parameters(), lr=self.lRate)
        nd_augmenter_optimizer = torch.optim.Adam(nd_augmenter.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            augloss, recloss, gnnloss = 0,0,0
            for n, batch in enumerate(next_batch_pairwisev2(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch   # [batch_size]
                '''
                # new loss
                # train augmenter
                
                augmenter.train()
                augmenter.zero_grad()
                model.eval()
                
                rec_user_emb, rec_item_emb = model()
                node_embs = torch.cat([rec_user_emb, rec_item_emb],0)
                
                perturbed_adj1, mean_aug_edge_weight1 = augmenter(node_embs, self.sparse_norm_adj)
                #perturbed_adj2, mean_aug_edge_weight2 = augmenter(node_embs, self.sparse_norm_adj)
                rec_user_emb1, rec_item_emb1 = model(perturbed_adj1)
                #rec_user_emb2, rec_item_emb2 = model(perturbed_adj2)

                user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb1[user_idx], rec_item_emb1[pos_idx], rec_item_emb1[neg_idx]
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                user_emb_cat = 1/2 * (user_emb + user_emb1)
                pos_item_emb_cat = 1/2 * (pos_item_emb + pos_item_emb1)
                neg_item_emb_cat = 1/2 * (neg_item_emb + neg_item_emb1)

                rec_loss1 = bpr_loss(user_emb1, pos_item_emb1, neg_item_emb1)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                rec_loss_both = bpr_loss(user_emb_cat, pos_item_emb_cat, neg_item_emb_cat)

                _IG1G2 = self.calc_loss([user_idx, pos_idx], rec_user_emb1, rec_item_emb1, rec_user_emb, rec_item_emb)
                _IG1G2Y = rec_loss + rec_loss1 - rec_loss_both

                aug_loss = self.cl_rate * _IG1G2 + (2-self.cl_rate) * _IG1G2Y
                #aug_loss += l2_reg_loss(self.reg, user_emb1, pos_item_emb1, neg_item_emb1)
                #aug_loss += l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)

                #regularition
                #l2_reg = 0
                #for param in augmenter.parameters():
                #    l2_reg += torch.norm(param)
                #l2_reg = self.reg * l2_reg

                # limit the ratio of droped edge
                reg_aug = self.reg_drop * mean_aug_edge_weight1
                #aug_loss += l2_reg
                aug_loss -= reg_aug
                
                aug_loss.backward()
                augmenter_optimizer.step()  

                # train model
                model.train()
                augmenter.eval()
                model.zero_grad()

                rec_user_emb, rec_item_emb = model()
                node_embs = torch.cat([rec_user_emb, rec_item_emb],0)
                
                perturbed_adj1, mean_aug_edge_weight1 = augmenter(node_embs, self.sparse_norm_adj)
                rec_user_emb1, rec_item_emb1 = model(perturbed_adj1)

                user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb1[user_idx], rec_item_emb1[pos_idx], rec_item_emb1[neg_idx]
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                user_emb_cat = 1/2 * (user_emb + user_emb1)
                pos_item_emb_cat = 1/2 * (pos_item_emb + pos_item_emb1)
                neg_item_emb_cat = 1/2 * (neg_item_emb + neg_item_emb1)

                rec_loss1 = bpr_loss(user_emb1, pos_item_emb1, neg_item_emb1)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                rec_loss_both = bpr_loss(user_emb_cat, pos_item_emb_cat, neg_item_emb_cat)

                _IG1G2 = self.calc_loss([user_idx, pos_idx], rec_user_emb1, rec_item_emb1, rec_user_emb, rec_item_emb)
                _IG1G2Y = rec_loss + rec_loss1 - rec_loss_both

                gnn_loss = self.cl_rate * _IG1G2 + (2-self.cl_rate) * _IG1G2Y
                #gnn_loss += l2_reg_loss(self.reg, user_emb1, pos_item_emb1, neg_item_emb1)
                gnn_loss += l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                
                gnn_loss.backward()
                model_optimizer.step()
                '''
                # train augmenter
                
                ed_augmenter.train()
                ed_augmenter.zero_grad()
                nd_augmenter.train()
                nd_augmenter.zero_grad()
                model.eval()
                
                rec_user_emb, rec_item_emb = model()
                
                #perturbed_adj1, mean_aug_edge_weight1 = ed_augmenter(rec_user_emb, rec_item_emb, self.sparse_norm_adj)
                #user_emb1, item_emb1 = model(perturbed_adj1)
                user_emb1, item_emb1 , mean_aug_edge_weight1 = nd_augmenter(rec_user_emb, rec_item_emb, self.sparse_norm_adj)
                user_emb2, item_emb2 , mean_aug_edge_weight2 = nd_augmenter(rec_user_emb, rec_item_emb, self.sparse_norm_adj)
                
                #user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb1[user_idx], rec_item_emb1[pos_idx], rec_item_emb1[neg_idx]
                #user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                #rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                cl_loss = - self.calc_loss([user_idx, pos_idx], user_emb1, item_emb1, user_emb2, item_emb2)

                #aug_loss += l2_reg_loss(self.reg, user_emb1, pos_item_emb1, neg_item_emb1)
                #aug_loss += l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)

                #regularition
                #l2_reg = 0
                #for param in augmenter.parameters():
                #    l2_reg += torch.norm(param)
                #l2_reg = self.reg * l2_reg
               
                # limit the ratio of droped edge
                #reg_aug = self.reg_drop * mean_aug_edge_weight1     
                
                aug_loss = cl_loss 
                aug_loss.backward()
                ed_augmenter_optimizer.step()
                nd_augmenter_optimizer.step()  

                # train model
                model.train()
                ed_augmenter.eval()
                nd_augmenter.eval()
                model.zero_grad()

                rec_user_emb, rec_item_emb = model()
                
                perturbed_adj1, mean_aug_edge_weight1 = ed_augmenter(rec_user_emb, rec_item_emb, self.sparse_norm_adj)
                user_emb1, item_emb1 = model(perturbed_adj1)
                user_emb2, item_emb2 , mean_aug_edge_weight2 = nd_augmenter(rec_user_emb, rec_item_emb, self.sparse_norm_adj)

                #user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb1[user_idx], rec_item_emb1[pos_idx], rec_item_emb1[neg_idx]
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                cl_loss = self.cl_rate * self.calc_loss([user_idx, pos_idx], user_emb1, item_emb1, user_emb2, item_emb2)

                gnn_loss = cl_loss + rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                
                gnn_loss.backward()
                model_optimizer.step()
                
                augloss += aug_loss.item()
                recloss += rec_loss.item()
                gnnloss += gnn_loss.item()

                if n % 100 == 0:
                    print('model training:', epoch + 1, 'batch', n, 'bpr_loss:', rec_loss.item(), 'cl_loss', cl_loss.item(), 'gnn_loss', gnn_loss.item())
                    print('augmenter training:', epoch + 1, 'batch', n, 'aug_loss', aug_loss.item())

            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                self.fast_evaluation(epoch)

            recall = self.performance['Recall']
            ndcg = self.performance['NDCG']
            augloss /= self.num_interaction
            recloss /= self.num_interaction
            gnnloss /= self.num_interaction
            self.writer.add_scalar('aug_loss', augloss, epoch)
            self.writer.add_scalar('rec_loss', recloss, epoch)
            self.writer.add_scalar('gnn_loss', gnnloss, epoch)
            self.writer.add_scalar('Recall', recall, epoch)
            self.writer.add_scalar('NDCG', ndcg, epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def calc_loss(self, idx, user_view_1, item_view_1, user_view_2, item_view_2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

class ADGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp):
        super(ADGCL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        #self.aug_type = aug_type
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
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
    
class BernEdgeAugmenter(torch.nn.Module):
    def __init__(self, emb_dim, b_temp, mlp_edge_model_dim=64):
        super(BernEdgeAugmenter, self).__init__()

        self.input_dim = emb_dim
        self.b_temp = b_temp
        #self.mlp_layer = mlp_layer

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, user_emb, item_emb, sp_tensor):
        # node_emb : [user_emb;item_emb]
        node_emb = torch.cat([user_emb, item_emb],0)
        inds = sp_tensor._indices()
        shape = sp_tensor.size()
        len_edges = inds.size()[1]
        vals = sp_tensor._values()
        
        #max_row = int(inds[0][len_edges//2-1])+1
        #max_col = int(shape[0]-max_row)
        src, dst = inds[0][:len_edges//2], inds[1][:len_edges//2]
        
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb) # dim : [num_of_edges//2 , 1]
        #print(inds.size())
        #print(edge_logits.shape)
        temperature = self.b_temp
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edge_logits) / temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()  # [num_of_edge//2]
        mean_edge_weight = torch.sum(aug_edge_weight)/(len_edges//2)
        #print(aug_edge_weight.shape)
    
        # edge_logits to new adj 
        n_inds = torch.LongTensor([src.tolist(), dst.tolist()]).cuda()
        num_nodes = int(shape[0])
        new_vals = torch.mul(vals[:len_edges//2], aug_edge_weight)
        with torch.no_grad():
            n_adj = torch.sparse.FloatTensor(n_inds, new_vals, shape)
        n_adj += torch.transpose(n_adj,0,1)
        #print(n_adj.size(), n_adj._indices().size())
        #pdb.set_trace()

        return n_adj, mean_edge_weight

    def forward1(self, node_emb, sp_tensor):
        # node_emb : [user_emb;item_emb]
        inds = sp_tensor._indices()
        shape = sp_tensor.size()
        len_edges = inds.size()[1]
        vals = sp_tensor._values()
        
        src, dst = inds[0][:], inds[1][:]
        
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb) # dim : [num_of_edges , 1]
       
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edge_logits) / temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()  # [num_of_edge]
        mean_edge_weight = torch.sum(aug_edge_weight)/(len_edges)
        #print(aug_edge_weight.shape)
    
        # edge_logits to new adj 
        new_vals = torch.mul(vals, aug_edge_weight)
        n_inds = torch.LongTensor([src.tolist(), dst.tolist()]).cuda()
        
        with torch.no_grad():
            n_adj = torch.sparse.FloatTensor(n_inds, new_vals, shape)       

        return n_adj, mean_edge_weight
    
class BernNodeAugmenter(torch.nn.Module):
    def __init__(self, emb_dim, b_temp, mlp_edge_model_dim=64):
        super(BernNodeAugmenter, self).__init__()

        self.input_dim = emb_dim
        self.b_temp = b_temp
        #self.mlp_layer = mlp_layer

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, user_emb, item_emb, sp_tensor):
        # node_emb : [user_emb;item_emb]
        node_emb = torch.cat([user_emb, item_emb],0)
        num_nodes = node_emb.shape[0]
        
        node_logits = self.mlp_edge_model(node_emb) # dim : [num_of_nodes , 1]
        #print(inds.size())
        #print(edge_logits.shape)
        temperature = self.b_temp
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(node_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + node_logits) / temperature
        aug_node_weight = torch.sigmoid(gate_inputs)  # [num_of_nodes, 1]
        mean_node_weight = torch.sum(aug_node_weight)/(num_nodes)
    
        # node dropout
        n_node_emb = torch.mul(node_emb, aug_node_weight)
        n_user_emb, n_item_emb = torch.split(n_node_emb, [user_emb.shape[0], item_emb.shape[0]], dim=0)

        return n_user_emb, n_item_emb, mean_node_weight
    

'''    
class BernMLPAugmenter1(torch.nn.Module):
    def __init__(self, emb_dim, n_layer, mlp_edge_model_dim=64):
        super(BernEdgeAugmenter, self).__init__()

        self.input_dim = emb_dim
        #self.mlp_layer = mlp_layer
        self.n_layer = n_layer

        self.mlp_edge_models = nn.ModuleList([Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1)
        ) for _ in range(self.n_layer)])
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, sp_tensor):
        # node_emb : [user_emb;item_emb]
        inds = sp_tensor._indices()
        shape = sp_tensor.size()
        len_edges = inds.size()[1]
        vals = sp_tensor._values()
        
        #max_row = int(inds[0][len_edges//2-1])+1
        #max_col = int(shape[0]-max_row)
        src, dst = inds[0][:len_edges//2], inds[1][:len_edges//2]
        
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)

        adjs, mean_logits = [], 0
        for i in range(self.n_layer):
            
            edge_logits = self.mlp_edge_models[i](edge_emb) # dim : [num_of_edges//2 , 1]
            #print(inds.size())
            #print(edge_logits.shape)
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + edge_logits) / temperature
            aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()  # [num_of_edge//2]
            mean_edge_weight = torch.sum(aug_edge_weight)/(len_edges//2)
            #print(aug_edge_weight.shape)
        
            # edge_logits to new adj 
            n_inds = torch.LongTensor([src.tolist(), dst.tolist()]).cuda()
            num_nodes = int(shape[0])
            new_vals = torch.mul(vals[:len_edges//2], aug_edge_weight)
            with torch.no_grad():
                n_adj = torch.sparse.FloatTensor(n_inds, new_vals, shape)
            n_adj += torch.transpose(n_adj,0,1)

            adjs.append(n_adj)
            mean_logits += mean_edge_weight

        return adjs, mean_logits/self.n_layer
'''