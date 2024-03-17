## reference: Qrec/base/iterativeRecommender.py
import os
from base.recommender import Recommender  # Recommender 主要是提取conf的参数
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation, ranking_split_evaluation, ranking_evaluation_group
import torch
import numpy as np
import sys
from util.conf import OptionConf


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.performance = None
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.earlyStop=0
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.dir = './log_process/' + self.config['model.name'] + '/'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.log_file = self.dir + current_time + '.txt'

        with open(self.log_file, 'a') as f:
            model_name = self.config['model.name']
            bp = model_name + ' conf: ' + self.config[model_name] + '\n'
            f.write(bp)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        """
        :return: dict[user] = list((item, scores)) topk of item and score
        """
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rGet test result Progress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
           
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8  # 去掉train_data 的 scores
           
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations result (itemId, ranking score, is hit).\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ','
                is_hit = '0'
                if item[0] in self.data.test_set[user]:
                    is_hit = '1'
                line += is_hit + ')'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.output['-dir']
        data_dir = self.config['training.set']
        dataset_name = data_dir.split('/')[-2]
        file_name = self.config['model.name'] + '_' + dataset_name + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['model.name']+ '_' + dataset_name + '@' + current_time + '-performance' + '.txt'

        self.result += ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def fast_evaluation(self, epoch):
        """record the best performance every epoch, method of identifying the best performance:
        when the number of better metrics is more than number of worse metrics for now epoch, now performance is better
        :return: list, [metric1:value1, ...]
        """
        print('evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        # self.bestPerformance [epoch, {metric:value}]
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
                self.earlyStop=0
            else:
                #stop model learning if the performance has not increased for 5 successive epochs
                self.earlyStop += 1
                '''
                if self.earlyStop==40:
                    model_name = self.config['model.name']
                    print('Early Stop at epoch', epoch+1)
                    bp = 'dataset: ' + self.config['training.set'] + '\n'
                    bp += self.config['item.ranking'] + '\n'
                    bp += model_name + ' conf: ' + self.config[model_name] + '\n'
                    bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + ' | '
                    bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + ' | '
                    bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
                    bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
                    print('*Best Performance* ')
                    print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
                    current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
                    file_name = self.config['model.name'] + '_' + self.config['training.set'].split('/')[-2] + '@' + current_time +'.txt'
                    out_put = self.output['-dir']
                    FileIO.write_file(out_put, file_name, bp)
                    exit(0)
                '''
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance.append(performance)
            self.save()
        self.performance = performance
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        file_name = self.config['model.name'] + '_' + self.config['training.set'].split('/')[-2] + '@' + current_time +'.txt'
        with open(self.log_file, 'a') as f:
            f.write(file_name)
            f.write('\n')
            contest = 'Epoch: {epoch}, Performance: {performance}'.format(epoch=epoch+1, performance=performance)
            f.write(contest)
            f.write('\n')
        f.close()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + ' | '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'MDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        
        return measure
    
    def group_eval(self,model):
        user_emb, item_emb = model()
        
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rGet test result Progress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()
            
        def predict(u, user_emb, item_emb):
            u = self.data.get_user_id(u)
            score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
            return score.cpu().numpy()

       
        # predict
        rec_list = {}
 
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            
            candidates = predict(user,user_emb,item_emb)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
           
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8  # 去掉train_data 的 scores
           
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        
        # eval
        file_name = self.config['model.name'] + '_splited_test_' + self.config['training.set'].split('/')[-2] +'.txt'
        filepath = self.dir + '_splited_test_' + self.config['training.set'].split('/')[-2] +'.txt'
        if os.path.exists(filepath):
            os.remove(filepath)
        
        measures = ranking_split_evaluation(self.data, rec_list, [self.max_N])
        for g,measure in enumerate(measures):
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            print('split eval: {performance}'.format(performance=performance))
        
            with open(filepath, 'a') as f:
                f.write(file_name)
                f.write('\n')
                contest = 'Group: {g}, Performance: {performance}'.format(g=g, performance=performance)
                f.write(contest)
                f.write('\n')
        f.close()
    
    def group_evaluation(self,model):
        
        
        if self.model_name == 'SIAB':
            user_emb, item_emb ,_,_ = model()
        else:
            user_emb, item_emb = model()
        
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rGet test result Progress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()
            
        def predict(u, user_emb, item_emb):
            u = self.data.get_user_id(u)
            score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
            return score.cpu().numpy()

       
        # predict
        recList = {}
 
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            
            candidates = predict(user,user_emb,item_emb)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
           
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8  # 去掉train_data 的 scores
           
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')

        #item grouping
        popularity = {}
        for item in self.data.training_set_i:
            popularity[item] = len(self.data.training_set_i[item])
        testset = []
        for user in self.data.test_set:
            for item in self.data.test_set[user]:
                if item in popularity:
                    testset.append((user,item,popularity[item]))
        testset = sorted(testset,key=lambda d:d[2])
        p_idx = np.linspace(0,len(testset),11)
        p_idx = [round(n) for n in p_idx]

        groups = []
        for i in range(10):
            group = {}
            for user in recList:
                group[user]={}
            for k in testset[p_idx[i]:p_idx[i + 1]]:
                group[k[0]][k[1]] = 1
            groups.append(group)


        measures=[]
        for group in groups:
            ms = ranking_evaluation_group(group, recList, self.data.test_set, self.max_N)
            measures.append(ms)

        print('###Group Evaluation - ITEM')
        print('model {m} group recalls:'.format(m=self.model_name))
        for r in measures:
            print(r)