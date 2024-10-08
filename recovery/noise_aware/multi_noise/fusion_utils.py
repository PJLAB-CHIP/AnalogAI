import logging
import copy
import numpy as np
import torch
import sys
import os
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn

from utils.utils import create_optimizer
from utils.earlystopping import EarlyStopping



class Client:
    def __init__(self, 
                 client_idx:int,
                 train_data:DataLoader, 
                 validation_data:DataLoader,
                 args, 
                 config,
                 device,
                 model_trainer,
                 model,
                 global_model):
        self.client_idx = client_idx
        self.train_data = train_data
        self.validation_data = validation_data
        self.global_model = global_model
        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.config = config
        self.optimizer = create_optimizer(model, 
                                        config.training.lr, 
                                        config.training.momentum, 
                                        config.training.weight_decay, 
                                        config.training.optimizer, 
                                        config.recovery.optimizer.sam, 
                                        config.recovery.optimizer.adaptive)

        self.pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=True)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=20, verbose=True)

    def get_sample_number(self):
        return len(self.train_data.dataset)

    def train(self, epochs=None):
        model, optimizer = self.model_trainer.train(
            client_idx = self.client_idx,
            model=self.model,
            global_model = self.global_model,
            criterion=self.criterion,
            optimizer = self.optimizer,
            train_data = self.train_data,
            validation_data = self.validation_data,
            config = self.config,
            pla_lr_scheduler = self.pla_lr_scheduler,
            device = self.device
        )
        weights = self.model_trainer.get_model_params(self.model)
        return weights,model,optimizer
        

    def local_test(self):
        test_data = self.validation_data
        metrics = self.model_trainer.test(self.model, test_data, self.device, self.args)
        return metrics


def agg_FedAvg(w_locals):
    '''
    FedAvg aggregation
    param w_locals: list of (sample_num, model_params)
    '''
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params

def compute_similarity(args, s_locals):
    '''
    compute the data distribution similarity of clients
    param s_locals: dict{layer_name: data distribution representation}
    return: list(client_num * client_num * similarity)
    '''
    client_num = len(s_locals)
    # similarities_dict = [[] for _ in range(client_num)] # dict{layer_name: similarity}
    similarities = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            # similarities_dict[i].append(dict())
            for k in s_locals[i]:
                if args.similarity_method == 0:
                    similarities[i][j] += torch.cosine_similarity(s_locals[i][k], s_locals[j][k], dim=0).item()
                elif args.similarity_method == 1:
                    similarities[i][j] += 1.0 / torch.dist(s_locals[i][k], s_locals[j][k], p=1).item()
                elif args.similarity_method == 2:
                    similarities[i][j] += 1.0 / torch.dist(s_locals[i][k], s_locals[j][k], p=2).item()
                else:
                    assert False
                    
    return similarities

def save_model(model, opt, epoch, save_file):
    print('==> Saving...')
    # state = {
    #     'opt': opt,
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'epoch': epoch,
    # }
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state