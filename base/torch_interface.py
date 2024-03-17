import torch
import torch_sparse

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        #print("sss")
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col]) 
        v = torch.from_numpy(coo.data).float()
        #print(torch.sparse.FloatTensor(i, v, coo.shape).cuda())
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    @staticmethod
    def convert_sparse_mat_to_coo_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)
    
    @staticmethod
    def convert_to_sparsetensor(X):
        coo = X.tocoo()
        row = torch.LongTensor(coo.row)
        col = torch.LongTensor(coo.col)
        value = torch.from_numpy(coo.data).float()
        shape = coo.shape
        spmat = torch_sparse.SparseTensor(row=row,col=col,value=value,sparse_sizes=shape)
        return spmat        