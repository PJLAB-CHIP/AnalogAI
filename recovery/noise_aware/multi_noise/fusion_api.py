from ast import arg
import copy
import logging
from logging import config
import os
import random

import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import wandb
from InferHardware.ibm_aihwkit import infer_aihwkit
import collections

from .fusion_utils import Client,agg_FedAvg,save_model
from utils.earlystopping import EarlyStopping

early_stopping = EarlyStopping(patience=20, verbose=True)

class FedProxAPI_personal(object):
    def __init__(self, 
                 train_data,
                 validation_data, 
                 device,
                 args,
                 config,
                 model_trainer, 
                 global_model, 
                 local_models):
        self.device = device
        self.args = args
        self.config = config
        self.global_model = global_model
        self.client_list = []
        self.model_trainer = model_trainer
        self.train_data = train_data
        self.validation_data = validation_data
        self._setup_clients(train_data, validation_data, model_trainer, local_models)
        
        # Loss
        self.client_test_acc = [[] for _ in range(len(self.client_list))]
        self.client_test_loss = [[] for _ in range(len(self.client_list))]
        self.avg_client_test_loss = []
        self.avg_client_test_acc = []

        self.global_test_loss = []
        self.global_test_acc = []
        

    def _setup_clients(self, train_data, validation_data, model_trainer, local_models):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx=client_idx,
                       train_data=train_data,
                       validation_data=validation_data,
                       args=self.args,
                       config=self.config,
                       device=self.device,
                       model_trainer=model_trainer,
                       model=local_models[client_idx],
                       global_model=self.global_model
                    )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        n_w = self.config.inference.platform.aihwkit.n_w # aihwkit weight
        save_dir=self.config.save_dir

        w_global = self.model_trainer.get_model_params(self.global_model) # global model weight
        # step1: init client model weight (using global weight)
        for idx, client in enumerate(self.client_list):
            self.model_trainer.set_model_params(client.model, w_global)
        
        acc_locals = []
        # step2: training
        for round_idx in range(1,self.args.comm_round+1):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = [] # record the weight of each client

            # (optional): weight selected method definition
            """for the first round"""
            if round_idx == 1:
                for _ in range(self.args.client_num_in_total):
                    acc_locals.append(1/self.args.client_num_in_total)

            # acc_locals = [0.0,0.05,0.10,0.15,0.20] # TODO: 临时使用加权平均

            # step2.1: train individually for each client
            for idx, client in enumerate(self.client_list):
                # (extra): get noise config for each client
                if idx == 0:
                    self.config.recovery.noise = self.config.recovery.noise_0
                elif idx == 1:
                    self.config.recovery.noise = self.config.recovery.noise_1
                elif idx == 2:
                    self.config.recovery.noise = self.config.recovery.noise_2
                elif idx == 3:
                    self.config.recovery.noise = self.config.recovery.noise_3
                elif idx == 4:
                    self.config.recovery.noise = self.config.recovery.noise_4
                elif idx == 5:
                    self.config.recovery.noise = self.config.recovery.noise_5
                elif idx == 6:
                    self.config.recovery.noise = self.config.recovery.noise_6
                elif idx == 7:
                    self.config.recovery.noise = self.config.recovery.noise_7
                elif idx == 8:
                    self.config.recovery.noise = self.config.recovery.noise_8
                elif idx == 9:
                    self.config.recovery.noise = self.config.recovery.noise_9

                w,model,optimizer = client.train()
                # w_locals.append((client.get_sample_number(), copy.deepcopy(w))) # avg using sample number
                print('acc_locals:', acc_locals)
                print('idx:', idx)
                w_locals.append((acc_locals[idx],copy.deepcopy(w)))
                
                # (optional) test results on the client model
                with torch.no_grad():
                    client.model, valid_loss, error, accuracy = self.model_trainer.test(
                        self.validation_data, client.model, nn.CrossEntropyLoss() , self.device
                    )
                if self.args.use_wandb:
                    pass
                    # wandb.log({'epoch':round_idx, 'accuracy_global':accuracy,'valid_loss_global':valid_loss})
                print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Round Index: {round_idx}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Test accuracy: {accuracy:.2f}%\t")

                if self.config.inference.platform.aihwkit.use:
                    print("==> inferencing on IBM(global)") 
                    infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(client.model)
                    _, _, error_aihwkit, accuracy_aihwkit = self.model_trainer.test(
                                    self.validation_data, infer_model_aihwkit, nn.CrossEntropyLoss(), self.device
                                )
                    print(f'Test accuracy aihwkit global: {accuracy_aihwkit}')

                client.early_stopping(
                    accuracy,
                    client.model.state_dict(),
                    idx,
                    self.config.data.architecture,
                    round_idx,
                    os.path.join(save_dir,f'client_{idx}_{self.config.recovery.noise.act_inject.sigma}_{self.config.recovery.noise.weight_inject.level}')
                )
            
            """(optional) online learning"""
            # acc_locals = [] # record acc tested on the analog computing platform
            # # (optional): test on the analog computing platform --> update client's weight
            # if self.config.inference.platform.aihwkit.use:
            #     for idx, client in enumerate(self.client_list):
            #         print("==> inferencing on IBM") 
            #         infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(client.model)
            #         _, _, error, accuracy = self.model_trainer.test(
            #                         self.validation_data, infer_model_aihwkit, nn.CrossEntropyLoss(), self.device
            #                     )
            #         if self.args.use_wandb:
            #             wandb.log({f'accuracy_aihwkit_{idx}':accuracy})
            #         print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')
            #         acc_locals.append(accuracy)

            if self.config.training.use_FI:
                # step2.2: update global weights and local weights
                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(self.global_model, w_global)
                for idx, client in enumerate(self.client_list):
                    if self.args.use_momentum:
                        _w_fused_model = collections.OrderedDict()
                        _w_client = self.model_trainer.get_model_params(client.model)
                        for k in _w_client.keys():
                            _v = 0.1 * _w_client[k] + 0.9 * w_global[k]
                            _w_fused_model.update({k:_v})
                        
                        self.model_trainer.set_model_params(client.model, _w_fused_model)
                    else:
                        self.model_trainer.set_model_params(client.model, w_global)
            
                # step2.3: test results on the global model
                with torch.no_grad():
                    self.global_model, valid_loss, error, accuracy = self.model_trainer.test(
                        self.validation_data, self.global_model, nn.CrossEntropyLoss() , self.device
                    )

                    if self.config.inference.platform.aihwkit.use:
                        print("==> inferencing on IBM(global)") 
                        infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(self.global_model)
                        _, _, error_aihwkit, accuracy_aihwkit = self.model_trainer.test(
                                        self.validation_data, infer_model_aihwkit, nn.CrossEntropyLoss(), self.device
                                    )
                        print(f'Test accuracy aihwkit global: {accuracy_aihwkit}')
                        if self.args.use_wandb:
                            wandb.log({'accuracy_aihwkit_global':accuracy_aihwkit})
                    if self.args.use_wandb:
                        wandb.log({'epoch':round_idx, 'accuracy_global':accuracy,'valid_loss_global':valid_loss})
                    print(
                    f"{datetime.now().time().replace(microsecond=0)} --- "
                    f"Round Index: {round_idx}\t"
                    f"Valid loss: {valid_loss:.4f}\t"
                    f"Test error: {error:.2f}%\t"
                    f"Test accuracy: {accuracy:.2f}%\t"
                )

                early_stopping(accuracy, 
                                self.global_model.state_dict(), 
                                -1,
                                self.config.data.architecture,
                                round_idx, 
                                save_dir)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        if self.args.use_wandb:
            wandb.finish()        

    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)