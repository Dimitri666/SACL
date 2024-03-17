from data.loader import FileIO
from util.conf import ModelConf
from util.rand import _set_random_seed
import os
import torch
import argparse
print(torch.cuda.device_count())
print(torch.version.cuda)

class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config

        self.training_data = FileIO.load_data_set(config['training.set']) # [[ui,ii,r],...]
        self.test_data = FileIO.load_data_set(config['test.set'])

        self.kwargs = {}

        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        print('***recommender start***')
        eval(recommender).execute()


if __name__ == '__main__':
    # added models
    all_models = ['SGL', 'SimGCL', 'DiffGCL', 'ADGCL', 'HCCL', 'ADGCL', 'CAFI', 'SIAB', 'NCL', 'MixGCF', 'DirectAU', 'LightGCN']

    print('=' * 80)
    
    
    parser = argparse.ArgumentParser(description='model name')
    
    parser.add_argument('-name', '--modelname', type=str, help='modelname')
    
    args = parser.parse_args()
    
    model = args.modelname
    import time

    s = time.time()
    if model in all_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    print(conf.config)
    seed = int(conf['seed'])
    _set_random_seed(seed)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))