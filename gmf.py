# -*- coding: UTF-8 -*-
import torch
from engine import Engine
from utils import use_cuda


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.num_ASnode1_info_type = config['num_ASnode1_info_type']
        self.num_ASnode2_info_type = config['num_ASnode2_info_type']
        self.num_ASnode1_AS_tier = config['num_ASnode1_AS_tier']
        self.num_ASnode2_AS_tier = config['num_ASnode2_AS_tier']
        self.num_ASnode1_info_traffic = config['num_ASnode1_info_traffic']
        self.num_ASnode2_info_traffic = config['num_ASnode2_info_traffic']
        self.num_ASnode1_info_ratio = config['num_ASnode1_info_ratio']
        self.num_ASnode2_info_ratio = config['num_ASnode2_info_ratio']
        self.num_ASnode1_info_scope = config['num_ASnode1_info_scope']
        self.num_ASnode2_info_scope = config['num_ASnode2_info_scope']
        self.num_ASnode1_policy_general = config['num_ASnode1_policy_general']
        self.num_ASnode2_policy_general = config['num_ASnode2_policy_general']
        self.num_ASnode1_policy_locations = config[
            'num_ASnode1_policy_locations']
        self.num_ASnode2_policy_locations = config[
            'num_ASnode2_policy_locations']
        self.num_ASnode1_policy_ratio = config['num_ASnode1_policy_ratio']
        self.num_ASnode2_policy_ratio = config['num_ASnode2_policy_ratio']
        self.num_ASnode1_policy_contracts = config[
            'num_ASnode1_policy_contracts']
        self.num_ASnode2_policy_contracts = config[
            'num_ASnode2_policy_contracts']

        self.num_ASnode1_appearIXP = config['num_ASnode1_appearIXP']
        self.num_ASnode2_appearIXP = config['num_ASnode2_appearIXP']
        self.num_ASnode1_appearFac = config['num_ASnode1_appearFac']
        self.num_ASnode2_appearFac = config['num_ASnode2_appearFac']



        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users,
                                                 embedding_dim=self.latent_dim)
        self.ASnode1_info_type = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_type,
            embedding_dim=self.num_ASnode1_info_type)
        self.ASnode1_AS_tier = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_AS_tier,
            embedding_dim=self.num_ASnode1_AS_tier)
        self.ASnode1_info_traffic = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_traffic,
            embedding_dim=self.num_ASnode1_info_traffic)
        self.ASnode1_info_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_ratio,
            embedding_dim=self.num_ASnode1_info_ratio)
        self.ASnode1_info_scope = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_scope,
            embedding_dim=self.num_ASnode1_info_scope)
        self.ASnode1_policy_general = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_general,
            embedding_dim=self.num_ASnode1_policy_general)
        self.ASnode1_policy_locations = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_locations,
            embedding_dim=self.num_ASnode1_policy_locations)
        self.ASnode1_policy_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_ratio,
            embedding_dim=self.num_ASnode1_policy_ratio)
        self.ASnode1_policy_contracts = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_contracts,
            embedding_dim=self.num_ASnode1_policy_contracts)
        self.ASnode1_appearIXP = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_appearIXP, embedding_dim=15)

        self.ASnode1_appearFac = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_appearFac, embedding_dim=20)


        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items,
                                                 embedding_dim=self.latent_dim)
        self.ASnode2_info_type = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_type,
            embedding_dim=self.num_ASnode2_info_type)
        self.ASnode2_AS_tier = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_AS_tier,
            embedding_dim=self.num_ASnode2_AS_tier)
        self.ASnode2_info_traffic = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_traffic,
            embedding_dim=self.num_ASnode2_info_traffic)
        self.ASnode2_info_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_ratio,
            embedding_dim=self.num_ASnode2_info_ratio)
        self.ASnode2_info_scope = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_scope,
            embedding_dim=self.num_ASnode2_info_scope)
        self.ASnode2_policy_general = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_general,
            embedding_dim=self.num_ASnode2_policy_general)
        self.ASnode2_policy_locations = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_locations,
            embedding_dim=self.num_ASnode2_policy_locations)
        self.ASnode2_policy_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_ratio,
            embedding_dim=self.num_ASnode2_policy_ratio)
        self.ASnode2_policy_contracts = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_contracts,
            embedding_dim=self.num_ASnode2_policy_contracts)
        self.ASnode2_appearIXP = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_appearIXP, embedding_dim=15)
        self.ASnode2_appearFac = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_appearFac, embedding_dim=20)



        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim + self.num_ASnode1_info_type +
            self.num_ASnode1_AS_tier + self.num_ASnode1_info_traffic +
            self.num_ASnode1_info_ratio + self.num_ASnode1_info_scope +
            self.num_ASnode1_policy_general +
            self.num_ASnode1_policy_locations + self.num_ASnode1_policy_ratio +
            self.num_ASnode1_policy_contracts+35 ,
            out_features=1)

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, ASnode1_info_type,
                ASnode1_AS_tier,  ASnode1_info_traffic,
                ASnode1_info_ratio, ASnode1_info_scope, ASnode1_policy_general,
                ASnode1_policy_locations, ASnode1_policy_ratio,
                ASnode1_policy_contracts,ASnode1_appearIXP, ASnode1_appearFac,
                ASnode2_info_type, ASnode2_AS_tier, ASnode2_info_traffic,
                ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general,
                ASnode2_policy_locations, ASnode2_policy_ratio,
                ASnode2_policy_contracts,ASnode2_appearIXP,
                ASnode2_appearFac):

        user_embedding = torch.cat(
            (self.embedding_user(user_indices),
             self.ASnode1_info_type(ASnode1_info_type),
             self.ASnode1_AS_tier(ASnode1_AS_tier),
             self.ASnode1_info_traffic(ASnode1_info_traffic),
             self.ASnode1_info_ratio(ASnode1_info_ratio),
             self.ASnode1_info_scope(ASnode1_info_scope),
             self.ASnode1_policy_general(ASnode1_policy_general),
             self.ASnode1_policy_locations(ASnode1_policy_locations),
             self.ASnode1_policy_ratio(ASnode1_policy_ratio),
             self.ASnode1_policy_contracts(ASnode1_policy_contracts),
             torch.mean(self.ASnode1_appearIXP(ASnode1_appearIXP), dim=1),
             torch.mean(self.ASnode1_appearFac(ASnode1_appearFac), dim=1)), 1)

        item_embedding = torch.cat(
            (self.embedding_item(item_indices),
             self.ASnode2_info_type(ASnode2_info_type),
             self.ASnode2_AS_tier(ASnode2_AS_tier),
             self.ASnode2_info_traffic(ASnode2_info_traffic),
             self.ASnode2_info_ratio(ASnode2_info_ratio),
             self.ASnode2_info_scope(ASnode2_info_scope),
             self.ASnode2_policy_general(ASnode2_policy_general),
             self.ASnode2_policy_locations(ASnode2_policy_locations),
             self.ASnode2_policy_ratio(ASnode2_policy_ratio),
             self.ASnode2_policy_contracts(ASnode2_policy_contracts),
             torch.mean(self.ASnode2_appearIXP(ASnode2_appearIXP), dim=1),
             torch.mean(self.ASnode2_appearFac(ASnode2_appearFac), dim=1)), 1)


        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = logits
        return rating

    def init_weight(self):
        pass


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)