# -*- coding: UTF-8 -*-
import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from ast import literal_eval
random.seed(0)
import numpy as np


def dealpadding(lis_str, max):
    list1 = np.zeros(max)
    eval_lis = literal_eval(lis_str)
    lis_len = len(eval_lis)
    for i in range(0, lis_len):
        list1[i] = int(eval_lis[i] + 1)
    return list1


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor,
                 ASnode1_info_type_tensor, ASnode1_AS_tier_tensor,
                 ASnode1_info_traffic_tensor, ASnode1_info_ratio_tensor,
                 ASnode1_info_scope_tensor, ASnode1_policy_general_tensor,
                 ASnode1_policy_locations_tensor, ASnode1_policy_ratio_tensor,
                 ASnode1_policy_contracts_tensor,ASnode1_appearIXP_tensor,
                 ASnode1_appearFac_tensor,
                 ASnode2_info_type_tensor, ASnode2_AS_tier_tensor,
                 ASnode2_info_traffic_tensor,ASnode2_info_ratio_tensor,
                 ASnode2_info_scope_tensor,ASnode2_policy_general_tensor,
                 ASnode2_policy_locations_tensor, ASnode2_policy_ratio_tensor,
                 ASnode2_policy_contracts_tensor,ASnode2_appearIXP_tensor,
                 ASnode2_appearFac_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.ASnode1_info_type_tensor = ASnode1_info_type_tensor
        self.ASnode1_AS_tier_tensor = ASnode1_AS_tier_tensor
        self.ASnode1_info_traffic_tensor = ASnode1_info_traffic_tensor
        self.ASnode1_info_ratio_tensor = ASnode1_info_ratio_tensor
        self.ASnode1_info_scope_tensor = ASnode1_info_scope_tensor
        self.ASnode1_policy_general_tensor = ASnode1_policy_general_tensor
        self.ASnode1_policy_locations_tensor = ASnode1_policy_locations_tensor
        self.ASnode1_policy_ratio_tensor = ASnode1_policy_ratio_tensor
        self.ASnode1_policy_contracts_tensor = ASnode1_policy_contracts_tensor
        self.ASnode1_appearIXP_tensor = ASnode1_appearIXP_tensor
        self.ASnode1_appearFac_tensor = ASnode1_appearFac_tensor
        self.ASnode2_info_type_tensor = ASnode2_info_type_tensor
        self.ASnode2_AS_tier_tensor = ASnode2_AS_tier_tensor
        self.ASnode2_info_traffic_tensor = ASnode2_info_traffic_tensor
        self.ASnode2_info_ratio_tensor = ASnode2_info_ratio_tensor
        self.ASnode2_info_scope_tensor = ASnode2_info_scope_tensor
        self.ASnode2_policy_general_tensor = ASnode2_policy_general_tensor
        self.ASnode2_policy_locations_tensor = ASnode2_policy_locations_tensor
        self.ASnode2_policy_ratio_tensor = ASnode2_policy_ratio_tensor
        self.ASnode2_policy_contracts_tensor = ASnode2_policy_contracts_tensor
        self.ASnode2_appearIXP_tensor = ASnode2_appearIXP_tensor
        self.ASnode2_appearFac_tensor = ASnode2_appearFac_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], \
               self.ASnode1_info_type_tensor[index], self.ASnode1_AS_tier_tensor[index],\
               self.ASnode1_info_traffic_tensor[index],self.ASnode1_info_ratio_tensor[index],\
               self.ASnode1_info_scope_tensor[index],self.ASnode1_policy_general_tensor[index],\
               self.ASnode1_policy_locations_tensor[index],self.ASnode1_policy_ratio_tensor[index],\
               self.ASnode1_policy_contracts_tensor[index],self.ASnode1_appearIXP_tensor[index] ,\
               self.ASnode1_appearFac_tensor[index],\
               self.ASnode2_info_type_tensor[index],\
               self.ASnode2_AS_tier_tensor[index], self.ASnode2_info_traffic_tensor[index],\
               self.ASnode2_info_ratio_tensor[index], self.ASnode2_info_scope_tensor[index],\
               self.ASnode2_policy_general_tensor[index],self.ASnode2_policy_locations_tensor[index],\
               self.ASnode2_policy_ratio_tensor[index], self.ASnode2_policy_contracts_tensor[index], \
               self.ASnode2_appearIXP_tensor[index], self.ASnode2_appearFac_tensor[index],self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""
    def __init__(self, train_ratings, valid_ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """

        self.train_ratings = train_ratings
        self.valid_ratings = valid_ratings


    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings,\
        ASnode1_info_type, ASnode1_AS_tier,ASnode1_info_traffic,ASnode1_info_ratio,ASnode1_info_scope,\
        ASnode1_policy_general,ASnode1_policy_locations,ASnode1_policy_ratio,ASnode1_policy_contracts, \
        ASnode1_appearIXP, ASnode1_appearFac,ASnode2_info_type,ASnode2_AS_tier,ASnode2_info_traffic,ASnode2_info_ratio,ASnode2_info_scope,\
        ASnode2_policy_general,ASnode2_policy_locations,ASnode2_policy_ratio,ASnode2_policy_contracts,ASnode2_appearIXP,ASnode2_appearFac= (
            [], [], [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])



        for row in self.train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            ASnode1_info_type.append(int(row.ASnode1_info_type))
            ASnode1_AS_tier.append(int(row.ASnode1_info_type))
            ASnode1_info_traffic.append(int(row.ASnode1_info_traffic))
            ASnode1_info_ratio.append(int(row.ASnode1_info_ratio))
            ASnode1_info_scope.append(int(row.ASnode1_info_scope))
            ASnode1_policy_general.append(int(row.ASnode1_policy_general))
            ASnode1_policy_locations.append(int(row.ASnode1_policy_locations))
            ASnode1_policy_ratio.append(int(row.ASnode1_policy_ratio))
            ASnode1_policy_contracts.append(int(row.ASnode1_policy_contracts))
            ASnode1_appearIXP.append(
                dealpadding(row.ASnode1_appearIXP, 879))
            ASnode1_appearFac.append(
                dealpadding(row.ASnode1_appearFac, 4111))



            ASnode2_info_type.append(int(row.ASnode2_info_type))
            ASnode2_AS_tier.append(int(row.ASnode2_info_type))
            ASnode2_info_traffic.append(int(row.ASnode2_info_traffic))
            ASnode2_info_ratio.append(int(row.ASnode2_info_ratio))
            ASnode2_info_scope.append(int(row.ASnode2_info_scope))
            ASnode2_policy_general.append(int(row.ASnode2_policy_general))
            ASnode2_policy_locations.append(int(row.ASnode2_policy_locations))
            ASnode2_policy_ratio.append(int(row.ASnode2_policy_ratio))
            ASnode2_policy_contracts.append(int(row.ASnode2_policy_contracts))
            ASnode2_appearIXP.append(
                dealpadding(row.ASnode2_appearIXP, 879))
            ASnode2_appearFac.append(
                dealpadding(row.ASnode2_appearFac, 4111))

            # construct data for model
        dataset = UserItemRatingDataset(
            user_tensor=torch.ShortTensor(users),
            item_tensor=torch.ShortTensor(items),
            target_tensor=torch.FloatTensor(ratings),
            ASnode1_info_type_tensor=torch.ShortTensor(ASnode1_info_type),
            ASnode1_AS_tier_tensor=torch.ShortTensor(ASnode1_AS_tier),
            ASnode1_info_traffic_tensor=torch.ShortTensor(ASnode1_info_traffic),
            ASnode1_info_ratio_tensor=torch.ShortTensor(ASnode1_info_ratio),
            ASnode1_info_scope_tensor=torch.ShortTensor(ASnode1_info_scope),
            ASnode1_policy_general_tensor=torch.ShortTensor(
                ASnode1_policy_general),
            ASnode1_policy_locations_tensor=torch.ShortTensor(
                ASnode1_policy_locations),
            ASnode1_policy_ratio_tensor=torch.ShortTensor(ASnode1_policy_ratio),
            ASnode1_policy_contracts_tensor=torch.ShortTensor(
                ASnode1_policy_contracts),
            ASnode1_appearIXP_tensor=torch.ShortTensor(ASnode1_appearIXP),
            ASnode1_appearFac_tensor=torch.ShortTensor(ASnode1_appearFac),
            ASnode2_info_type_tensor=torch.ShortTensor(ASnode2_info_type),
            ASnode2_AS_tier_tensor=torch.ShortTensor(ASnode2_AS_tier),
            ASnode2_info_traffic_tensor=torch.ShortTensor(ASnode2_info_traffic),
            ASnode2_info_ratio_tensor=torch.ShortTensor(ASnode2_info_ratio),
            ASnode2_info_scope_tensor=torch.ShortTensor(ASnode2_info_scope),
            ASnode2_policy_general_tensor=torch.ShortTensor(
                ASnode2_policy_general),
            ASnode2_policy_locations_tensor=torch.ShortTensor(
                ASnode2_policy_locations),
            ASnode2_policy_ratio_tensor=torch.ShortTensor(ASnode2_policy_ratio),
            ASnode2_policy_contracts_tensor=torch.ShortTensor(
                ASnode2_policy_contracts),
            ASnode2_appearIXP_tensor = torch.ShortTensor(ASnode2_appearIXP),
            ASnode2_appearFac_tensor = torch.ShortTensor(ASnode2_appearFac))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ##得到验证集,现在的验证集要变成回归。
    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_users, test_items, test_ratings, ASnode1_info_type, \
        ASnode1_AS_tier, \
        ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope, \
        ASnode1_policy_general, ASnode1_policy_locations, \
        ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP, ASnode1_appearFac,\
        ASnode2_info_type, ASnode2_AS_tier,  \
        ASnode2_info_traffic, ASnode2_info_ratio, ASnode2_info_scope, \
        ASnode2_policy_general, ASnode2_policy_locations, \
        ASnode2_policy_ratio, ASnode2_policy_contracts,ASnode2_appearIXP, ASnode2_appearFac  = (
           [],[],[],[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        )



        for row in self.valid_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            test_ratings.append(float(row.rating))
            ASnode1_info_type.append(int(row.ASnode1_info_type))
            ASnode1_AS_tier.append(int(row.ASnode1_info_type))
            ASnode1_info_traffic.append(int(row.ASnode1_info_traffic))
            ASnode1_info_ratio.append(int(row.ASnode1_info_ratio))
            ASnode1_info_scope.append(int(row.ASnode1_info_scope))
            ASnode1_policy_general.append(int(row.ASnode1_policy_general))
            ASnode1_policy_locations.append(int(row.ASnode1_policy_locations))
            ASnode1_policy_ratio.append(int(row.ASnode1_policy_ratio))
            ASnode1_policy_contracts.append(int(row.ASnode1_policy_contracts))
            ASnode1_appearIXP.append(
                dealpadding(row.ASnode1_appearIXP, 879))
            ASnode1_appearFac.append(
                dealpadding(row.ASnode1_appearFac, 4111))


            ASnode2_info_type.append(int(row.ASnode2_info_type))
            ASnode2_AS_tier.append(int(row.ASnode2_info_type))
            ASnode2_info_traffic.append(int(row.ASnode2_info_traffic))
            ASnode2_info_ratio.append(int(row.ASnode2_info_ratio))
            ASnode2_info_scope.append(int(row.ASnode2_info_scope))
            ASnode2_policy_general.append(int(row.ASnode2_policy_general))
            ASnode2_policy_locations.append(int(row.ASnode2_policy_locations))
            ASnode2_policy_ratio.append(int(row.ASnode2_policy_ratio))
            ASnode2_policy_contracts.append(int(row.ASnode2_policy_contracts))
            ASnode2_appearIXP.append(
                dealpadding(row.ASnode2_appearIXP, 879))
            ASnode2_appearFac.append(
                dealpadding(row.ASnode2_appearFac, 4111))

        # print(torch.eye(len(ASnode2_info_prefixes4)))
        return [
            torch.ShortTensor(test_users),
            torch.ShortTensor(test_items),
            torch.ShortTensor(ASnode1_info_type),
            torch.ShortTensor(ASnode1_AS_tier),
            torch.ShortTensor(ASnode1_info_traffic),
            torch.ShortTensor(ASnode1_info_ratio),
            torch.ShortTensor(ASnode1_info_scope),
            torch.ShortTensor(ASnode1_policy_general),
            torch.ShortTensor(ASnode1_policy_locations),
            torch.ShortTensor(ASnode1_policy_ratio),
            torch.ShortTensor(ASnode1_policy_contracts),
            torch.ShortTensor(ASnode1_appearIXP),
            torch.ShortTensor(ASnode1_appearFac),
            torch.ShortTensor(ASnode2_info_type),
            torch.ShortTensor(ASnode2_AS_tier),
            torch.ShortTensor(ASnode2_info_traffic),
            torch.ShortTensor(ASnode2_info_ratio),
            torch.ShortTensor(ASnode2_info_scope),
            torch.ShortTensor(ASnode2_policy_general),
            torch.ShortTensor(ASnode2_policy_locations),
            torch.ShortTensor(ASnode2_policy_ratio),
            torch.ShortTensor(ASnode2_policy_contracts),
            torch.ShortTensor(ASnode2_appearIXP),
            torch.ShortTensor(ASnode2_appearFac),
            torch.FloatTensor(test_ratings)
        ]
