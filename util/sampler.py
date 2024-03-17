from random import shuffle, randint, choice


# training_data:[[user_id, item_id, float(weight)]...]


def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pairwise1(data, batch_size, n_negs=1):
    """
    1. shuffle 2.batch split 3.sample negative
    args:
        data
        batch_sieze
        n_negs = 64


    return:
        u_idx : [batch_size]
        i_idx : [batch_size]
        j_idx : [batch_size]

    """
    training_data = data.training_data
    # 传进来的参数data是data.ui_graph下的Interaction类
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:

        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())  # 以列表的方式来获取所有item_id

        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])  # batch_size
            u_idx.append(data.user[user])  # batch_size
            #neg_idx = []
            #for m in range(n_negs):
            neg_item = choice(item_list)
            while neg_item in data.training_set_u[user]:
                neg_item = choice(item_list)
            neg_idx = data.item[neg_item]
                # 这里的nef_item_id的shape应该是[batch,64]
            j_idx.append(neg_idx)
        yield u_idx, i_idx, j_idx


def next_batch_pairwisev2(data, batch_size, n_negs=1):
    """
    this is like in SGL-torch pairwisesampler, 1. sample neg_id; 2. shuffle 3. batch_split
    args:
        data
        batch_sieze
        n_negs = 64


    return:
        u_idx : [batch_size]
        i_idx : [batch_size]
        j_idx : [batch_size]

    """
    training_data = data.training_data
    # 传进来的参数data是data.ui_graph下的Interaction类
    users = [sublist[0] for sublist in training_data]
    items = [sublist[1] for sublist in training_data]
    all_ids = []
    # sampling negative
    item_list = list(data.item.keys()) 
    for i, user in enumerate(users):
        neg_item = choice(item_list)
        while neg_item in data.training_set_u[user]:
            neg_item = choice(item_list)
        all_ids.append([data.user[user], data.item[items[i]], data.item[neg_item]])

    shuffle(all_ids)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:

        if batch_id + batch_size <= data_size:
            u_idx = [all_ids[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            i_idx = [all_ids[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            j_idx = [all_ids[idx][2] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            u_idx = [all_ids[idx][0] for idx in range(batch_id, data_size)]
            i_idx = [all_ids[idx][1] for idx in range(batch_id, data_size)]
            j_idx = [all_ids[idx][2] for idx in range(batch_id, data_size)]
            batch_id = data_size
        
        yield u_idx, i_idx, j_idx
        
def next_batch_pairwisev3(data, batch_size, n_negs=1):
    """
    1. shuffle 2.batch split 3.sample negative
    args:
        data
        batch_sieze
        n_negs = 64


    return:
        u_idx : [batch_size]
        i_idx : [batch_size]
        j_idx : [batch_size]

    """
    training_data = data.training_data
    # 传进来的参数data是data.ui_graph下的Interaction类
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:

        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())  # 以列表的方式来获取所有item_id

        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])  # batch_size
            u_idx.append(data.user[user])  # batch_size
            #neg_idx = []
            #for m in range(n_negs):
            neg_item = choice(item_list)
            while neg_item in data.training_set_u[user]:
                neg_item = choice(item_list)
            neg_idx = data.item[neg_item]
                # 这里的nef_item_id的shape应该是[batch,64]
            j_idx.append(neg_idx)
            
            user_pop = data.user_pop_idx[data.user[user]]
            pos_item_pop = data.item_pop_idx[data.item[items[i]]]
            pos_weight = data.weights[data.item[items[i]]]
            neg_items_pop = data.item_pop_idx[data.item[items[neg_item]]]
            
        yield u_idx, i_idx, j_idx, user_pop, pos_item_pop, pos_weight, neg_items_pop


def next_batch_pointwise(data, batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y
