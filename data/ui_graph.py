import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph # 主要对邻接矩阵标准化 normalize_graph_mat
import scipy.sparse as sp
import pickle

class Interaction(Data,Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self,conf,training,test)
        ## 注意：这里和SGL不一致，SGL没有转id的操作，认为最大的user就是所有user的数量。此处只考虑存在的user,应该是大多数情况

        self.user = {}  # dict, 将存在的user从0开始编号,eg. 1：0， 2：1， 5：2，...
        self.item = {}  # 同self.user
        self.id2user = {} # dict, 和self.user相反，编号对应usereg. 0:1, 1:2, 2:5, ...
        self.id2item = {} # 同self.id2user
        self.training_set_u = defaultdict(dict) # 嵌套dict, 外层user对应多个item:rating
        self.training_set_i = defaultdict(dict) # 嵌套dict, 外层item对应多个user:rating
        self.train_user_list = defaultdict(dict)
        self.train_item_list = defaultdict(dict)
        
        self.test_set = defaultdict(dict) # 嵌套dict， 外层user是train_data中的，对应多个test_data中的item:rating
        self.grouped_test_set = []
        self.test_set_item = set() # set, 在train_data中的user对应的所有item集合
        self.__generate_set() # 生成上面的数据
        self.user_num = len(self.training_set_u) # 嵌套dict的len，只看外层的数量，即train_data中user的数量
        self.item_num = len(self.training_set_i) # train_data中item的数量
        
        #self.test_item_num = len(self.test_set_item)
        #self.n_items = self.item_num + self.test_item_num
        self.ui_adj = self.__create_sparse_bipartite_adjacency() # 构建邻接矩阵，注意这里矩阵的索引和原数据的索引不一致
        self.norm_adj = self.normalize_graph_mat(self.ui_adj) # 标准化邻接矩阵
        self.interaction_mat = self.__create_sparse_interaction_matrix() # 交互矩阵，非方阵
        self.norm_interation_mat = self.normalize_interaction_mat(self.interaction_mat)
        
        # popularity_user = {}
        # for u in self.user:
        #     popularity_user[self.user[u]] = len(self.training_set_u[u])
        # popularity_item = {}
        # for u in self.item:
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])
        self.neibs_users = self.neighbors_of_users()
        self.neibs_items = self.neighbors_of_items()
        
        self.population_list = []
        self.pop_dict = {}
        self.user_pop_idx = []
        self.item_pop_idx = []
        self.user_pop_max = 0
        self.item_pop_max = 0
        self.weights = []
        self._load_traindata()
        #self._split_group()

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                self.train_user_list[self.user[user]] = [item]
            else:
                self.train_user_list[self.user[user]].append(item)
                
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                self.train_item_list[self.item[item]] = [user]
            else:
                self.train_item_list[self.item[item]].append(user)
                
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
                    
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)
            
    def _load_traindata(self):
        # Population matrix
        self.pop_dict = {}
        for item, users in self.train_item_list.items():
            self.pop_dict[item] = len(users) + 1

            self.population_list.append(self.pop_dict[item])
        '''
        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in self.train_item_list.items()}
        self.pop_item = pop_item
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)
        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i
        self.user_pop_idx = np.zeros(self.user_num, dtype=int)
        self.item_pop_idx = np.zeros(self.item_num, dtype=int)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]

        self.user_pop_max = max(self.user_pop_idx)
        self.item_pop_max = max(self.item_pop_idx)
        self.weights = self.get_weight()
        '''
    
    def _split_group(self):
        max_pop = max(self.population_list)
        
        low, mid, high = int(1/3*max_pop), int(2/3*max_pop), int(max_pop)
        
        low_group = defaultdict(dict)
        mid_group = defaultdict(dict)
        high_group = defaultdict(dict)
        
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user:
                continue
            
            if item not in self.item.keys():
                continue
            
            if self.pop_dict[self.item[item]] <= low:
                low_group[user][item] = rating
            elif low < self.pop_dict[self.item[item]] <= mid:
                mid_group[user][item] = rating
            else:
                high_group[user][item] = rating
    
        self.grouped_test_set = [low_group, mid_group, high_group]
        
        
    def get_weight(self):
        pop = self.population_list
        pop = np.clip(pop, 1, max(pop))
        pop = pop / max(pop)
        return pop

        '''
        pop = self.population_list
        pop = np.clip(pop, 1, max(pop))
        pop = pop / np.linalg.norm(pop, ord=np.inf)
        pop = 1 / pop

        if 'c' in self.IPStype:
            pop = np.clip(pop, 1, np.median(pop))
        if 'n' in self.IPStype:
            pop = pop / np.linalg.norm(pop, ord=np.inf)

        return pop
        '''
        
    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        """
        :param adj_mat: 非方阵的邻接矩阵，shape:num_user, num_item
        :return: 变为方阵， shape:num_user, num_item
        """
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        """get id of user"""
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        """get id of item"""
        if i in self.item:
            return self.item[i]

    def training_size(self):
        """
        :return: num.user, num.item, num.training_data of train_data
        """
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        """
        :return: num.user, num.item, num.testing_data of in test_data
        """
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        """whether user u rated item i in train_data"""
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        """whether user is in training set"""
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        """
        :param u: user in training data
        :return: all items rated with u, all ratings between all rated items and u in training data
        """
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        """
        :param i: item in training data
        :return: all users rated with i, all ratings between all rated users and i in training data
        """
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        """
        :param u: user id in interaction matrix (shape: num_user, num_item)(not laplacian matrix)
        :return: uth row array in interaction matrix
        """
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        """
        :param i: item id in interaction matrix (shape: num_user, num_item)(not laplacian matrix)
        :return: ith column array in interaction matrix
        """
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def neighbors_of_u(self,u):
        row = self.row(u)
        return np.nonzero(row)

    def neighbors_of_i(self,i):
        col = self.col(i)
        return np.nonzero(col)
    
    def neighbors_of_users(self):
        neighbors_of_users = []
        u_ids = [x for x in range(self.user_num)]
        for u in u_ids:
            neighbors = self.neighbors_of_u(u)
            neighbors_of_users.append(neighbors)
        return neighbors_of_users

    def neighbors_of_items(self):
        neighbors_of_items = []
        i_ids = [x for x in range(self.item_num)]
        for i in i_ids:
            neighbors = self.neighbors_of_i(i)
            neighbors_of_items.append(neighbors)
        return neighbors_of_items

    def matrix(self):
        """
        :return: dense interaction array with shape (num_user, num_item)
        """
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
