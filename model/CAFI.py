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

class CAFI(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CAFI, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CAFI'])
        self.alpha = float(args['-alpha'])
        self.beta = float(args['-beta'])
        b_temp = float(args['-b_temp'])
        self.reg_drop = float(args['-reg_drop'])
        mlp_emb_size = int(args['-mlp_emb_size'])
        n_layers = int(args['-n_layer'])
        self.temp = float(args['-temp'])
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda() 
        self.num_interaction = int(self.sparse_norm_adj._values().size()[0])
        current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime(time()))
        self.writer = SummaryWriter(log_dir=(out_dir + '/' + self.model_name + '/' + dataset_name + '/' + current_time))
        self.model = CAFI_Encoder(self.data, self.emb_size, n_layers, self.temp, b_temp, mlp_emb_size)
        #self.augmenter = BernMLPAugmenter(self.emb_size, b_temp)

    def train(self):
        model = self.model.cuda()
        model_optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            recloss, clloss, aloss = 0,0,0
            model.train()
            for n, batch in enumerate(next_batch_pairwisev2(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch   # [batch_size]
                
                rec_user_emb, rec_item_emb, new_rec_user_emb, new_rec_item_emb, mask_mean = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                user_emb1, pos_item_emb1, neg_item_emb1 = new_rec_user_emb[user_idx], new_rec_item_emb[pos_idx], new_rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                rec_loss1 = bpr_loss(user_emb1, pos_item_emb1, neg_item_emb1)
                cl_loss = self.calc_loss([user_idx, pos_idx], rec_user_emb, rec_item_emb, new_rec_user_emb, new_rec_item_emb)            
                reg = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)

                # regularition of mask to eliminate drop too many 
                #reg_mask = self.reg_drop * mask_mean

                loss = rec_loss + self.beta * rec_loss1 + self.alpha * cl_loss + reg # - reg_mask

                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()

                recloss += rec_loss.item()
                clloss += cl_loss.item()
                aloss += loss.item()

                if n % 100 == 0:
                    print('model training:', epoch + 1, 'batch', n, 'bpr_loss:', rec_loss.item(), 'cl_loss', cl_loss.item(), 'batch_loss', loss.item())

            model.eval()
            with torch.no_grad():
                _, _, self.user_emb, self.item_emb, _ = self.model()
                self.fast_evaluation(epoch)

            recall = self.performance['Recall']
            ndcg = self.performance['NDCG']
            recloss /= self.num_interaction
            clloss /= self.num_interaction
            aloss /= self.num_interaction
            self.writer.add_scalar('rec_loss', recloss, epoch)
            self.writer.add_scalar('cl_loss', clloss, epoch)
            self.writer.add_scalar('gnn_loss', aloss, epoch)
            self.writer.add_scalar('Recall', recall, epoch)
            self.writer.add_scalar('NDCG', ndcg, epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            _, _, self.best_user_emb, self.best_item_emb, _ = self.model.forward()

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

class CAFI_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, temp, bt, mlp_edge_model_dim=64):
        super(CAFI_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.bern_temp = bt
        #self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()  
        self.embedding_dict = self._init_model()
        self.augmenter = nn.ModuleList([Sequential(Linear(self.emb_size, mlp_edge_model_dim), ReLU(), Linear(mlp_edge_model_dim, self.emb_size)) for _ in range(self.n_layers)])
        self.init_mlp()
        
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict
    
    def init_mlp(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        all_perturbed_embeddings = []
        mask_mean = 0
        for k in range(self.n_layers):
            # cal f(G)
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            # cal feature mask
            feat_logits = self.augmenter[k](ego_embeddings)

            temperature = self.bern_temp
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(feat_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + feat_logits) / temperature
            feat_mask = torch.sigmoid(gate_inputs).squeeze() #[num_nodes, emb_size]
            mask_mean += torch.mean(feat_logits)
            # cal masked feature t(f(G))
            perturbed_embeddings = torch.mul(ego_embeddings, feat_mask)
            
            all_perturbed_embeddings.append(perturbed_embeddings)
            ego_embeddings = perturbed_embeddings

        #  mean pooling f(G)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        # mean pooling t(f(G))
        all_perturbed_embeddings = torch.stack(all_perturbed_embeddings, dim=1)
        all_perturbed_embeddings = torch.mean(all_perturbed_embeddings, dim=1)
        user_all_perturbed_embeddings, item_all_perturbed_embeddings = torch.split(all_perturbed_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings, user_all_perturbed_embeddings, item_all_perturbed_embeddings, mask_mean
    
class BernMLPAugmenter(torch.nn.Module):
    def __init__(self, emb_dim, b_temp, mlp_edge_model_dim=64):
        super(BernMLPAugmenter, self).__init__()

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
    
class BernMLPAugmenter1(torch.nn.Module):
    def __init__(self, emb_dim, n_layer, mlp_edge_model_dim=64):
        super(BernMLPAugmenter, self).__init__()

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