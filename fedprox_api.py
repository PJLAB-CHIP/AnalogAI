from ast import arg
import copy
import logging
import os
import random

import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import wandb
from call_inference import infer_aihwkit

from fedprox_utils import Client,agg_FedAvg,save_model
from earlystopping import EarlyStopping

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
        """
        dataset: unlabel dataset
        train_label_data_local_dict: label dataset
        """
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

        # logging settings
        # TODO: 修改日志文件路径
        # self.formatted_time = args.formatted_time
        # if not os.path.exists(f'model/{self.formatted_time}'):
        #     os.mkdir(f'model/{self.formatted_time}')
        # logging.basicConfig(filename=f'model/{self.formatted_time}/log.log',level=logging.INFO)
        

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
        w_global = self.model_trainer.get_model_params(self.global_model) # global model weight
        # step1: init client model weight (using global weight)
        for idx, client in enumerate(self.client_list):
            self.model_trainer.set_model_params(client.model, w_global)
        
        # step2: training
        for round_idx in range(1,self.args.comm_round+1):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = [] # record the weight of each client

            # (optional): weight selected method definition
            acc_locals = [] # record acc tested on the analog computing platform
            """for the first round"""
            if round_idx == 1:
                for _ in range(self.args.client_num_in_total):
                    acc_locals.append(1/self.args.client_num_in_total)

            # step2.1: train individually for each client
            for idx, client in enumerate(self.client_list):
                w,model,optimizer = client.train()
                # w_locals.append((client.get_sample_number(), copy.deepcopy(w))) # avg using sample number

                # (extra): get noise config for each client
                # if idx == 0:
                #     self.config.recovery.noise = self.config.recovery.noise_0
                # elif idx == 1:
                #     self.config.recovery.noise = self.config.recovery.noise_1
                # elif idx == 2:
                #     self.config.recovery.noise = self.config.recovery.noise_2
                # elif idx == 3:
                #     self.config.recovery.noise = self.config.recovery.noise_3
                # elif idx == 4:
                #     self.config.recovery.noise = self.config.recovery.noise_4
                # w_locals.append((self.config.recovery.noise.act_inject.sigma, copy.deepcopy(w)))
                print('acc_locals:', acc_locals)
                print('idx:', idx)
                w_locals.append((acc_locals[idx],copy.deepcopy(w)))
            
            # step2.2: update global weights and local weights
            w_global = self._aggregate(w_locals)

            # (optional): test on the analog computing platform --> update client's weight
            if self.config.inference.platform.aihwkit:
                for idx, client in enumerate(self.client_list):
                    print("==> inferencing on IBM") 
                    n_w = 0.02
                    infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(client.model)
                    _, _, error, accuracy = self.model_trainer.test(
                                    self.validation_data, infer_model_aihwkit, nn.CrossEntropyLoss(), self.device
                                )
                    print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')
                    acc_locals.append(accuracy)
        
            self.model_trainer.set_model_params(self.global_model, w_global)
            for idx, client in enumerate(self.client_list):
                self.model_trainer.set_model_params(client.model, w_global)
            
            # step2.3: test results on the global model
            with torch.no_grad():
                self.global_model, valid_loss, error, accuracy = self.model_trainer.test(
                    self.validation_data, self.global_model, nn.CrossEntropyLoss() , self.device
                )
                wandb.log({'epoch':round_idx, 'accuracy_global':accuracy,'valid_loss_global':valid_loss})
                print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Round Index: {round_idx}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Test accuracy: {accuracy:.2f}%\t"
            )
            
            save_dir=self.config.save_dir
            best_accuracy = early_stopping(accuracy, 
                                        self.global_model.state_dict(), 
                                        -1,
                                        self.config.data.architecture,
                                        round_idx, 
                                        save_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        wandb.finish()
            


            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            #     self._global_test(round_idx)

            # elif round_idx % self.args.frequency_of_the_test == 0:
            #     self._local_test_on_all_clients(round_idx)
            #     self._global_test(round_idx)
            
            # TODO: 修改模型保存
            # if round_idx%self.args.save_freq==0:
            #     if self.args.use_fl:
            #         # step2.4: save model
            #         save_model(model=self.model_global,opt=self.args,epoch=round_idx,save_file=f'model/{self.formatted_time}/{round_idx}.pth')
            #     else:
            #         for i in range(len(self.client_list)):
            #             if not os.path.exists(f'model/{self.formatted_time}/client_{i}'):
            #                 os.mkdir(f'model/{self.formatted_time}/client_{i}')
            #             save_model(model=self.client_list[i].model,opt=self.args,epoch=round_idx,save_file=f'model/{self.formatted_time}/client_{i}/{round_idx}.pth')
        
        # logging.info('avg_client_test_acc = {}'.format(self.avg_client_test_acc))
        # logging.info('avg_client_test_loss = {}'.format(self.avg_client_test_loss))
        # logging.info('global_test_acc = {}'.format(self.global_test_acc))
        # logging.info('global_test_loss = {}'.format(self.global_test_loss))
        # logging.info('client_test_acc = {}'.format(self.client_test_acc))
        # logging.info('client_test_loss = {}'.format(self.client_test_loss))
        
       

    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)

    def _global_test(self, round_idx):
        
        logging.info("################global_test : {}".format(round_idx))

        test_acc = self.model_trainer.test(self.global_model, self.test_global, self.device, self.args)
        
        stats = {'global_test_acc': test_acc}

        logging.info(stats)
        
        self.global_test_acc.append(test_acc)
        # self.global_test_loss.append(test_loss)

    def _local_test_on_all_clients(self, round_idx):
        # TODO: 修改local test的机制
        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        local_test_acc_list = []
        local_test_loss_list = []

        for client_idx in range(self.args.client_num_in_total):
            
            # test data
            # test_local_metrics = client.local_test(True)
            test_acc, y_test, y_pre = self.client_list[client_idx].local_test()
            from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, accuracy_score, f1_score
            from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer         
            
            precisionScore = round(precision_score(y_test, y_pre, average='weighted'), 4)
            recallScore = round(recall_score(y_test, y_pre, average='weighted'),4)
            f1Score = round(f1_score(y_test, y_pre, average='weighted'),4)
            accuracy = round(accuracy_score(y_test, y_pre),4)
            
            # TODO: 暂时不测试ROC
            score = 0.0
            # score = roc_auc_score(y_pre, y_test)
            # score = round(score, 4)
            
            # test on test dataset
            # test_acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            # test_loss = test_local_metrics['test_loss'] / test_local_metrics['test_total']

            self.client_test_acc[client_idx].append(test_acc)
            # self.client_test_loss[client_idx].append(test_loss)

            local_test_acc_list.append(test_acc)
            info_client = {'client_id':client_idx, 'local_test_acc': test_acc, '预测精确率为':precisionScore, '预测准确率为': accuracy, '预测f1-score为': f1Score, '预测AUC为' : score, '预测召回率为': recallScore}
            print('info_client:', info_client)
            logging.info(info_client)
            # local_test_loss_list.append(test_loss)

        avg_local_test_acc = sum(local_test_acc_list) / len(local_test_acc_list)
        # avg_local_test_loss = sum(local_test_loss_list) / len(local_test_loss_list)

        stats = {'avg_local_test_acc': avg_local_test_acc}
        # wandb.log({"Avg Local Test/Acc": avg_local_test_acc, "round": round_idx})
        # wandb.log({"Avg Local Test/Loss": avg_local_test_loss, "round": round_idx})
        logging.info(stats)
        
        self.avg_client_test_acc.append(avg_local_test_acc)
        # self.avg_client_test_loss.append(avg_local_test_loss)