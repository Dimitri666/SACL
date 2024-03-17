import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        """
        方阵：D^-1/2 A D^-1/2
        非方阵：D^-1 A
        :param adj_mat:
        :return:
        """
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:   
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat
    
    @staticmethod
    def normalize_interaction_mat(adj_mat):
        """
        param: interaction mat 
        return: D(u)^-1/2 A D(i)^-1/2
        """
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        colsum = np.array(adj_mat.sum(0))

        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)

        c_inv = np.power(colsum, -0.5).flatten()
        c_inv[np.isinf(c_inv)] = 0.
        c_mat_inv = sp.diags(c_inv)

        norm_int_mat = r_mat_inv.dot(adj_mat)
        norm_int_mat = norm_int_mat.dot(c_mat_inv)
        return norm_int_mat


    def convert_to_laplacian_mat(self, adj_mat):
        pass
