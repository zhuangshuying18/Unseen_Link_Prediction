# -*- coding: UTF-8 -*-

from multiprocessing import  Process
import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator



def train(sample_generator,evaluate_data,gmf_config):
    print('in')
    config = gmf_config
    engine = GMFEngine(config)
    for epoch in range(config['num_epoch']):
        # print('Epoch {} starts !'.format(epoch))
        # print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        rmse = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, rmse)


if __name__ == '__main__':
    AShop_dir = './data/AShopmatrix/metrix_trainattribute.csv'
    AShop_train_rating = pd.read_csv(AShop_dir, sep=' ', header=None, names=['userId', 'itemId', 'rating',
        'ASnode1_info_type','ASnode1_AS_tier','ASnode1_info_traffic',
        'ASnode1_info_ratio','ASnode1_info_scope','ASnode1_policy_general', 'ASnode1_policy_locations', 'ASnode1_policy_ratio',
        'ASnode1_policy_contracts','ASnode1_appearIXP','ASnode1_appearFac',
        'ASnode2_info_type', 'ASnode2_AS_tier', 'ASnode2_info_traffic',
        'ASnode2_info_ratio', 'ASnode2_info_scope', 'ASnode2_policy_general', 'ASnode2_policy_locations', 'ASnode2_policy_ratio',
        'ASnode2_policy_contracts','ASnode2_appearIXP', 'ASnode2_appearFac'],    engine='python')
    AShop_train_rating.sort_values(by=["userId", "itemId"], inplace=True)
    AShop_train_rating.reset_index(drop=True, inplace=True)

    print(AShop_train_rating)

    AShop_valid_dir = './data/AShopmatrix/metrix_validattribute.csv'
    AShop_valid_rating = pd.read_csv(AShop_valid_dir, sep=' ', header=None, names=['userId', 'itemId', 'rating',
        'ASnode1_info_type','ASnode1_AS_tier','ASnode1_info_traffic',
        'ASnode1_info_ratio','ASnode1_info_scope','ASnode1_policy_general', 'ASnode1_policy_locations', 'ASnode1_policy_ratio',
        'ASnode1_policy_contracts','ASnode1_appearIXP','ASnode1_appearFac',
        'ASnode2_info_type', 'ASnode2_AS_tier', 'ASnode2_info_traffic',
        'ASnode2_info_ratio', 'ASnode2_info_scope', 'ASnode2_policy_general', 'ASnode2_policy_locations', 'ASnode2_policy_ratio',
        'ASnode2_policy_contracts' ,'ASnode2_appearIXP', 'ASnode2_appearFac'],    engine='python')
    AShop_valid_rating.sort_values(by=["userId", "itemId"], inplace=True)
    AShop_valid_rating.reset_index(drop=True, inplace=True)
    print(AShop_valid_rating)
    sample_generator = SampleGenerator(train_ratings=AShop_train_rating, valid_ratings=AShop_valid_rating)
    evaluate_data = sample_generator.evaluate_data






    gmf_config=({'alias': 'gmf_implict_nfactors' + str(100) + 'nepochs' + str(
                        20) + 'nbatch' + str(128) + 'lr' + str(0.001),
                                  'num_epoch': 20,
                                  'batch_size': 128,
                                  'optimizer': 'adam',
                                  'adam_lr': 0.001,
                                  'num_users': 72248,
                                  'num_items': 72248,
                                  'num_ASnode1_info_type':4,
                                  'num_ASnode2_info_type':4,
                                   'num_ASnode1_AS_tier': 5,
                                   'num_ASnode2_AS_tier': 5,
                                   'num_ASnode1_info_traffic': 19,
                                                 'num_ASnode2_info_traffic': 19,
                                                 'num_ASnode1_info_ratio': 6,
                                                 'num_ASnode2_info_ratio': 6,
                                                 'num_ASnode1_info_scope': 10,
                                                 'num_ASnode2_info_scope': 10,
                                                 'num_ASnode1_policy_general': 4,
                                                 'num_ASnode2_policy_general': 4,
                                                 'num_ASnode1_policy_locations': 6,
                                                 'num_ASnode2_policy_locations': 6,
                                                 'num_ASnode1_policy_ratio': 3,
                                                 'num_ASnode2_policy_ratio': 3,
                                                 'num_ASnode1_policy_contracts': 4,
                                                 'num_ASnode2_policy_contracts': 4,

                                                 'num_ASnode1_appearIXP': 879,
                                                 'num_ASnode2_appearIXP': 879,
                                                 'num_ASnode1_appearFac': 4111,
                                                 'num_ASnode2_appearFac': 4111,


                                  'latent_dim': 100,
                                  'l2_regularization': 0,  # 0.01
                                  'use_cuda': True,
                                  'device_id': 8,
                                  'model_dir': 'checkpoints/gmf/nfactors' + str(100) + 'nepochs' + str(
                                      20) + 'nbatch' + str(128) + 'lr' + str(
                                      0.001) + '{}_Epoch{}_RMSE{:.4f}.model'})





    train(sample_generator,evaluate_data,gmf_config)

