# -*- coding: UTF-8 -*-
import torch
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK
import numpy as np


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """
    def __init__(self, config):
        self.config = config  # model configuration
        self._writer = SummaryWriter(log_dir='runs/{}'.format(
            config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # loss function
        self.crit = torch.nn.MSELoss()

    def train_single_batch(
            self, users, items, ASnode1_info_type, ASnode1_AS_tier,

            ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
            ASnode1_policy_general, ASnode1_policy_locations,
            ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP,
            ASnode1_appearFac, ASnode2_info_type, ASnode2_AS_tier,
            ASnode2_info_traffic, ASnode2_info_ratio, ASnode2_info_scope,
            ASnode2_policy_general, ASnode2_policy_locations,
            ASnode2_policy_ratio, ASnode2_policy_contracts, ASnode2_appearIXP,
            ASnode2_appearFac, ratings):


        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # move to GPU
        if self.config['use_cuda'] is True:
            users, items,ASnode1_info_type, ASnode1_AS_tier,ASnode1_info_traffic,ASnode1_info_ratio,ASnode1_info_scope,\
            ASnode1_policy_general,ASnode1_policy_locations,ASnode1_policy_ratio,\
            ASnode1_policy_contracts,ASnode1_appearIXP,ASnode1_appearFac,ASnode2_info_type,\
            ASnode2_AS_tier,ASnode2_info_traffic,\
            ASnode2_info_ratio,ASnode2_info_scope,ASnode2_policy_general,ASnode2_policy_locations,\
            ASnode2_policy_ratio,ASnode2_policy_contracts,ASnode2_appearIXP,ASnode2_appearFac, ratings = \
            users.cuda(), items.cuda(), ASnode1_info_type.cuda(), ASnode1_AS_tier.cuda(),\
            ASnode1_info_traffic.cuda(),\
            ASnode1_info_ratio.cuda(),ASnode1_info_scope.cuda(),ASnode1_policy_general.cuda(),\
            ASnode1_policy_locations.cuda(),ASnode1_policy_ratio.cuda(),ASnode1_policy_contracts.cuda(), \
            ASnode1_appearIXP.cuda(), ASnode1_appearFac.cuda(),ASnode2_info_type.cuda(),\
            ASnode2_AS_tier.cuda(),\
            ASnode2_info_traffic.cuda(),ASnode2_info_ratio.cuda(),ASnode2_info_scope.cuda(),\
            ASnode2_policy_general.cuda(),ASnode2_policy_locations.cuda() ,ASnode2_policy_ratio.cuda(),\
            ASnode2_policy_contracts.cuda(), ASnode2_appearIXP.cuda(),ASnode2_appearFac.cuda(),ratings.cuda()

        self.opt.zero_grad()


        ratings_pred = self.model(
            users, items, ASnode1_info_type, ASnode1_AS_tier,
            ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
            ASnode1_policy_general, ASnode1_policy_locations,
            ASnode1_policy_ratio, ASnode1_policy_contracts,ASnode1_appearIXP,
            ASnode1_appearFac,  ASnode2_info_type, ASnode2_AS_tier,
            ASnode2_info_traffic, ASnode2_info_ratio, ASnode2_info_scope,
            ASnode2_policy_general, ASnode2_policy_locations,
            ASnode2_policy_ratio, ASnode2_policy_contracts,ASnode2_appearIXP,
            ASnode2_appearFac)
        # calculate loss,
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):

        assert hasattr(self, 'model'), 'Please specify the exact model !'
        #
        self.model.train()
        total_loss = 0
        #
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.ShortTensor)
            #
            user, item, \
            ASnode1_info_type, ASnode1_AS_tier,ASnode1_info_traffic,ASnode1_info_ratio,ASnode1_info_scope,ASnode1_policy_general,ASnode1_policy_locations,\
            ASnode1_policy_ratio,ASnode1_policy_contracts,ASnode1_appearIXP,ASnode1_appearFac,\
            ASnode2_info_type, ASnode2_AS_tier,ASnode2_info_traffic,ASnode2_info_ratio,ASnode2_info_scope,ASnode2_policy_general,ASnode2_policy_locations,\
            ASnode2_policy_ratio,ASnode2_policy_contracts,ASnode2_appearIXP,ASnode2_appearFac,rating = (
                batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8],
                batch[9],batch[10],batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17],
                batch[18], batch[19],batch[20],batch[21],batch[22],batch[23], batch[24])

            rating = rating.float()
            loss = self.train_single_batch(
                user, item, ASnode1_info_type, ASnode1_AS_tier,
                ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
                ASnode1_policy_general, ASnode1_policy_locations,
                ASnode1_policy_ratio, ASnode1_policy_contracts,     ASnode1_appearIXP, ASnode1_appearFac,
                ASnode2_info_type,
                ASnode2_AS_tier, ASnode2_info_traffic,
                ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general,
                ASnode2_policy_locations, ASnode2_policy_ratio,
                ASnode2_policy_contracts, ASnode2_appearIXP, ASnode2_appearFac,rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(
                epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items,ASnode1_info_type, ASnode1_AS_tier,ASnode1_info_traffic,\
            ASnode1_info_ratio,ASnode1_info_scope,ASnode1_policy_general,\
            ASnode1_policy_locations,ASnode1_policy_ratio,ASnode1_policy_contracts, \
            ASnode1_appearIXP, ASnode1_appearFac,ASnode2_info_type, ASnode2_AS_tier,ASnode2_info_traffic,\
            ASnode2_info_ratio,ASnode2_info_scope,ASnode2_policy_general,\
            ASnode2_policy_locations,ASnode2_policy_ratio,ASnode2_policy_contracts,    ASnode2_appearIXP,ASnode2_appearFac,\
            test_ratings= (
                evaluate_data[0], evaluate_data[1],evaluate_data[2],evaluate_data[3],
                evaluate_data[4],evaluate_data[5], evaluate_data[6], evaluate_data[7],
                evaluate_data[8],evaluate_data[9], evaluate_data[10],evaluate_data[11],
                evaluate_data[12], evaluate_data[13], evaluate_data[14], evaluate_data[15],
                evaluate_data[16], evaluate_data[17], evaluate_data[18], evaluate_data[19],
                evaluate_data[20],evaluate_data[21], evaluate_data[22], evaluate_data[23],
                evaluate_data[24])

            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                test_ratings = test_ratings.cuda()
                ASnode1_info_type = ASnode1_info_type.cuda()
                ASnode1_AS_tier = ASnode1_AS_tier.cuda()
                ASnode1_info_traffic = ASnode1_info_traffic.cuda()
                ASnode1_info_ratio = ASnode1_info_ratio.cuda()
                ASnode1_info_scope = ASnode1_info_scope.cuda()
                ASnode1_policy_general = ASnode1_policy_general.cuda()
                ASnode1_policy_locations = ASnode1_policy_locations.cuda()
                ASnode1_policy_ratio = ASnode1_policy_ratio.cuda()
                ASnode1_policy_contracts = ASnode1_policy_contracts.cuda()
                ASnode1_appearIXP = ASnode1_appearIXP.cuda()
                ASnode1_appearFac = ASnode1_appearFac.cuda()
                ASnode2_info_type = ASnode2_info_type.cuda()
                ASnode2_AS_tier = ASnode2_AS_tier.cuda()
                ASnode2_info_traffic = ASnode2_info_traffic.cuda()
                ASnode2_info_ratio = ASnode2_info_ratio.cuda()
                ASnode2_info_scope = ASnode2_info_scope.cuda()
                ASnode2_policy_general = ASnode2_policy_general.cuda()
                ASnode2_policy_locations = ASnode2_policy_locations.cuda()
                ASnode2_policy_ratio = ASnode2_policy_ratio.cuda()
                ASnode2_policy_contracts = ASnode2_policy_contracts.cuda()
                ASnode2_appearIXP = ASnode2_appearIXP.cuda()
                ASnode2_appearFac = ASnode2_appearFac.cuda()

            test_scores = self.model(
                test_users, test_items, ASnode1_info_type, ASnode1_AS_tier,
                ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
                ASnode1_policy_general, ASnode1_policy_locations,
                ASnode1_policy_ratio, ASnode1_policy_contracts,  ASnode1_appearIXP, ASnode1_appearFac,
                ASnode2_info_type,
                ASnode2_AS_tier, ASnode2_info_traffic,
                ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general,
                ASnode2_policy_locations, ASnode2_policy_ratio,
                ASnode2_policy_contracts,ASnode2_appearIXP, ASnode2_appearFac)
        rmse = np.sqrt(
            mean_squared_error(y_pred=test_scores.detach().cpu().numpy(),
                               y_true=test_ratings.detach().cpu().numpy()))
        self._writer.add_scalar('performance/RMSE', rmse, epoch_id)
        print('[Evluating Epoch {}] RMSE = {:.4f}'.format(epoch_id, rmse))
        return rmse

    def save(self, alias, epoch_id, rmse):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, rmse)
        save_checkpoint(self.model, model_dir)