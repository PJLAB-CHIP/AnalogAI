import logging
from datetime import datetime

from typing import Optional
import torch
from torch import nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch import max as torch_max

from tqdm import tqdm
import math
import wandb

from qat.fake_quantize import fake_quantize_prepare
from recovery.noise_aware.noise_inject import InjectForward, InjectWeight, InjectWeightNoise
from utils.utils import train_step, test_evaluation
from utils.earlystopping import EarlyStopping

early_stopping = EarlyStopping(patience=20, verbose=True)

class FedProxTrainer(object):
    
    def get_model_params(self, model):
        return copy.deepcopy(model.state_dict())

    def set_model_params(self, model, model_parameters):
        model.load_state_dict(copy.deepcopy(model_parameters))

    def train(self, 
              client_idx,
              model:nn.Module,
              global_model:nn.Module,
              criterion, 
              optimizer, 
              train_data, 
              validation_data,
              config, 
              pla_lr_scheduler,
              device,
              print_every=1
              ) -> float:
        """_summary_

        Args:
            model (nn.Module): Trained model to be evaluated
            criterion (nn.CrossEntropyLoss): criterion to compute loss
            optimizer (Optimizer): analog model optimizer
            train_data (DataLoader): Validation set to perform the evaluation
            validation_data (DataLoader): Validation set to perform the evaluation
            epochs (int): global parameter to define epochs number
            print_every (int): defines how many times to print training progress

        Returns:
            float: _description_
        """
        save_dir = config.save_dir
        train_losses = []
        valid_losses = []
        test_error = []

        if config.recovery.qat.use:
            model = fake_quantize_prepare(model=model, 
                                        device=device, 
                                        a_bits=config.recovery.qat.a_bits, 
                                        w_bits=config.recovery.qat.w_bits, )
            
        # (extra): get noise config for each client
        if client_idx == 0:
            config.recovery.noise = config.recovery.noise_0
        elif client_idx == 1:
            config.recovery.noise = config.recovery.noise_1
        elif client_idx == 2:
            config.recovery.noise = config.recovery.noise_2
        elif client_idx == 3:
            config.recovery.noise = config.recovery.noise_3
        elif client_idx == 4:
            config.recovery.noise = config.recovery.noise_4
        elif client_idx == 5:
            config.recovery.noise = config.recovery.noise_5
        elif client_idx == 6:
            config.recovery.noise = config.recovery.noise_6
        elif client_idx == 7:
            config.recovery.noise = config.recovery.noise_7
        elif client_idx == 8:
            config.recovery.noise = config.recovery.noise_8
        elif client_idx == 9:
            config.recovery.noise = config.recovery.noise_9
        # print('@@@@----->',model)    
        print(config.recovery.noise)
        # Train model
        for epoch in range(0, config.training.epochs):
            # Train_step
            if  config.recovery.noise.act_inject.use:
                print('====>inject forward noise<====')
                noise_a = InjectForward(fault_type=config.recovery.noise.act_inject.type, 
                                        arg1=config.recovery.noise.act_inject.mean, 
                                        arg2=config.recovery.noise.act_inject.sigma,
                                        sigma_global=0.25,
                                        sigma_dict=None,
                                        layer_mask=config.recovery.noise.act_inject.mask)
            else:
                noise_a = None
                
            if  config.recovery.noise.weight_inject.use:
                print('====>inject weight noise<====')
                noise_w = InjectWeightNoise(model, 
                                        noise_level=config.recovery.noise.weight_inject.level)
            else:
                noise_w = None
            model, optimizer, train_loss = train_step(train_data, 
                                                    model, 
                                                    global_model,
                                                    criterion, 
                                                    optimizer, 
                                                    device, 
                                                    config,
                                                    noise_a,
                                                    noise_w)
            train_losses.append(train_loss)

            if epoch % print_every == (print_every - 1):
                # Validate_step
                with torch.no_grad():
                    model, valid_loss, error, accuracy = test_evaluation(
                        validation_data, model, criterion, device
                    )
                    valid_losses.append(valid_loss)
                    test_error.append(error)

                print(
                    f"{datetime.now().time().replace(microsecond=0)} --- "
                    f"Epoch: {epoch}\t"
                    f"Train loss: {train_loss:.4f}\t"
                    f"Valid loss: {valid_loss:.4f}\t"
                    f"Test error: {error:.2f}%\t"
                    f"Test accuracy: {accuracy:.2f}%\t"
                )  
            pla_lr_scheduler.step(valid_loss)   
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if config.use_wandb:
                wandb.log({f'accuracy_{client_idx}':accuracy, f'train_loss_{client_idx}':train_loss, f'valid_loss_{client_idx}':valid_loss})
                # wandb.log({'epoch':epoch, f'accuracy_{client_idx}':accuracy, f'train_loss_{client_idx}':train_loss, f'valid_loss_{client_idx}':valid_loss})
        return model, optimizer
    
    def test(self,
             validation_data,
             model,
             criterion, device):
        """Test trained network

        Args:
            validation_data (DataLoader): Validation set to perform the evaluation
            model (nn.Module): Trained model to be evaluated
            criterion (nn.CrossEntropyLoss): criterion to compute loss

        Returns:
            nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
        """
        total_loss = 0
        predicted_ok = 0
        total_images = 0

        model.eval()
        criterion.to(device)
        model.to(device)

        # for images, labels in validation_data:
        t = tqdm(validation_data, leave=False, total=len(validation_data))
        for _, (images, labels) in enumerate(t):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
            accuracy = predicted_ok / total_images * 100
            error = (1 - predicted_ok / total_images) * 100

        epoch_loss = total_loss / len(validation_data.dataset)
        return model, epoch_loss, error, accuracy
