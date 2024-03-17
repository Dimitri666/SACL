import numpy as np
import random
import scipy.sparse as sp

class GraphAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def node_dropout(sp_adj, drop_rate):
        """Input: a sparse adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj
   

    @staticmethod
    def edge_adadrop(sp_adj, drop_rate, reserve_rate, reverse=False):
        """drop these edges of lowest connect possibility"""
        # normalize the coo interaction matrix
        colsum = np.array(sp_adj.sum(0))
        rowsum = np.array(sp_adj.sum(1))

        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)

        c_inv = np.power(colsum, -0.5).flatten()
        c_inv[np.isinf(c_inv)] = 0.
        c_mat_inv = sp.diags(c_inv)

        normalized_sp_mat = (r_mat_inv.dot(sp_adj)).dot(c_mat_inv)
        normalized_sp_mat = normalized_sp_mat.tocoo()

        # sort and dropout
        shape = sp_adj.shape
        edge_count = sp_adj.count_nonzero()
        reserve_edge = int(edge_count * reserve_rate)
        keep_idx = random.sample(range(edge_count-reserve_edge), int((edge_count-reserve_edge) * (1 - drop_rate)))
        if reverse:
            sorted_indices = np.argsort(-normalized_sp_mat.data)
        else:
            sorted_indices = np.argsort(normalized_sp_mat.data)
        sorted_data = sp_adj.data[sorted_indices]
        sorted_row = normalized_sp_mat.row[sorted_indices]
        sorted_col = normalized_sp_mat.col[sorted_indices]

        dropped_data = sorted_data[keep_idx]
        dropped_row = sorted_row[keep_idx]
        dropped_col = sorted_col[keep_idx]

        dropped_adj = sp.csr_matrix((dropped_data,(dropped_row, dropped_col)), shape=shape)

        return dropped_adj