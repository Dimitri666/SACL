## reference: Qrec/util/measure.py

import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        """
        :param origin: data.test_set
        :param res: rec_list
        :return: dict, user: number of items both in test_set and rec_list for this user
        """
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            if user in res.keys():
                predicted = [item[0] for item in res[user]]
                hit_count[user] = len(set(items).intersection(set(predicted)))
            else:
                hit_count[user] = 0
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved(hit) interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num/total_num

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        """
        :param hits: dict, user:hit_num
        :param N: int, topN
        :return: float, sum of hit_num of all users  / num_users * N
        """
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        """
        :param hits: dict, user:hit_num
        :param origin: data.test_set
        :return: mean of (num_hit / num_items in test) of all users
        """
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

    # @staticmethod
    # def MAP(origin, res, N):
    #     sum_prec = 0
    #     for user in res:
    #         hits = 0
    #         precision = 0
    #         for n, item in enumerate(res[user]):
    #             if item[0] in origin[user]:
    #                 hits += 1
    #                 precision += hits / (n + 1.0)
    #         sum_prec += precision / min(len(origin[user]), N)
    #     return sum_prec / len(res)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)


def ranking_evaluation(origin, res, N):
    """
    calculate hit_num, hit_ratio, precision@n, recall@n, ndcg@n
    :param origin: data.test_set {user:{item:rating}}
    :param res: rec_list {user:(item,score)}
    :param N: [10,20]
    :return: list, [[hit ratios\n, precision\n, recall\n, ndcg\n, n=10\n],[..., n=20\n]]
    """
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure


def ranking_evaluation_group(group, recList, test_set, max_N):
    """
    calculate hit_num, hit_ratio, precision@n, recall@n, ndcg@n
    :param origin: data.test_set {user:{item:rating}}
    :param res: rec_list {user:(item,score)}
    :param N: [10,20]
    :return: list, [[hit ratios\n, precision\n, recall\n, ndcg\n, n=10\n],[..., n=20\n]]
    """
    measure = []
    recall_list = []
    for user in group:
        if user in recList:
            # Get recommended items for the user
            recommended_items = {item for item, _ in recList[user]}

            # Get ground truth items for the user from the test set
            true_items = set(test_set[user].keys())

            # group items 
            group_items = set(group[user].keys())
            
            # hit items in group 
            hits_group = group_items.intersection(recommended_items)
            
            # Calculate precision, recall, and F1 score
            
            recall = len(hits_group.intersection(true_items)) / len(true_items) if true_items else 0.0
            
            recall_list.append(recall)
            
    recall_all = sum(recall_list) / len(recall_list)
            
    return recall_all


def ranking_split_evaluation(data, res, N):
    """
    calculate hit_num, hit_ratio, precision@n, recall@n, ndcg@n
    :param origin: data.test_set {user:{item:rating}}
    :param group: grouped test_set {user:{item:rating}}
    :param res: rec_list {user:(item,score)}
    :param N: [10,20]
    :return: list, [[hit ratios\n, precision\n, recall\n, ndcg\n, n=10\n],[..., n=20\n]]
    """
    measure = []
    measure1 = []
    measure2 = []
    measure3 = []
    origin = data.test_set
    pop_dict = data.pop_dict
    max_pop = max(pop_dict.values())
    low, mid, high = int(1/3*max_pop), int(2/3*max_pop), int(max_pop)
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of group set and predicted set do not match!')
            exit(-1)
            
        ## all 
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
        
        ## low 
        low_predicted = {user:[(item,score) for item, score in items if pop_dict[data.item[item]] <= low] for user, items in predicted.items()}
        low_hits = Metric.hits(origin, low_predicted)
        hr = Metric.hit_ratio(origin, low_hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(low_hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(low_hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, low_predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure1.append('Top ' + str(n) + '\n')
        measure1 += indicators 
        
        # mid 
        mid_predicted = {user:[(item,score) for item, score in items if low<pop_dict[data.item[item]] <= mid] for user, items in predicted.items()}
        mid_hits = Metric.hits(origin, mid_predicted)
        hr = Metric.hit_ratio(origin, mid_hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(mid_hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(mid_hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, mid_predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure2.append('Top ' + str(n) + '\n')
        measure2 += indicators 
        
        # high
        high_predicted = {user:[(item,score) for item, score in items if mid < pop_dict[data.item[item]]] for user, items in predicted.items()}
        high_hits = Metric.hits(origin, high_predicted)
        hr = Metric.hit_ratio(origin, high_hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(high_hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(high_hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, high_predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure3.append('Top ' + str(n) + '\n')
        measure3 += indicators 
        
    return measure, measure1, measure2, measure3


def rating_evaluation(res):
    """calculating MAE and RMSE"""
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure