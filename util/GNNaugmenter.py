import torch
from torch.nn import Sequential, Linear, ReLU
import numpy as np
import pdb

class BernMLPAugmenter(torch.nn.Module):
    def __init__(self, emb_dim, mlp_edge_model_dim=64):
        super(BernMLPAugmenter, self).__init__()

        self.input_dim = emb_dim

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
        
        #max_row = int(inds[0][len_edges//2-1])+1
        #max_col = int(shape[0]-max_row)
        src, dst = inds[0][:len_edges//2-1], inds[1][:len_edges//2-1]
        
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb) # dim : [num_of_edges//2 , 1]
        #print(inds.size())
        #print(edge_logits.shape)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edge_logits) / temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()  # [num_of_edge//2]
        #print(aug_edge_weight.shape)
    
        # edge_logits to new adj 
        n_inds = torch.LongTensor([src.tolist(), dst.tolist()]).cuda()
        num_nodes = int(shape[0])
        n_adj = torch.sparse.FloatTensor(n_inds, aug_edge_weight, shape)
        n_adj += torch.transpose(n_adj,0,1)
        print(n_adj.size(), n_adj._indices().size())
        #pdb.set_trace()

        return n_adj, aug_edge_weight
    
class EMA:
    """
    Exponential Moving Average (EMA) work for siamese network
    """
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new

        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new
    
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)