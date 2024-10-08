# -*- coding: utf-8 -*-
"""
version: 1.0
train.py
"""

# Imports
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn, device, no_grad, save
from .model import resnet, vgg, lenet, mlp 
from utils.utils import dict2namespace, create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from data.dataset import load_dataset
from utils.earlystopping import EarlyStopping
# FedProx
from recovery.noise_aware.multi_noise.model_trainer_fusion import FedProxTrainer
from recovery.noise_aware.multi_noise.fusion_api import FedProxAPI_personal
from args import parse_option

import argparse
import yaml
import wandb

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

args = parse_option()

if args.use_wandb:
    wandb.login()

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

device = config.training.device

# Path to store datasets
data_dir = os.path.join(os.getcwd(), "data", config.data.dataset)

# Path to store results
# basic_dir = (args.config).replace('.yml','')

"""Normalize the experiment setup"""
min_noise_intensity = config.recovery.noise_0.act_inject.sigma
_min_noise_intensity = config.recovery.noise_0.weight_inject.level
if config.training.client_num_in_total == 5 or config.training.client_num_in_total == 1:
    max_noise_intensity = config.recovery.noise_4.act_inject.sigma
    _max_noise_intensity = config.recovery.noise_4.weight_inject.level
elif config.training.client_num_in_total == 10:
    max_noise_intensity = config.recovery.noise_9.act_inject.sigma
    _max_noise_intensity = config.recovery.noise_9.weight_inject.level

EXP_BASIC = True # TODO:
if EXP_BASIC:
    basic_dir = (args.config).split('.')[0]
else:
    basic_dir = f'{config.data.architecture}_{config.data.dataset}_client_{config.training.client_num_in_total}_epoch_{config.training.epochs}_{config.recovery.noise_0.act_inject.use}_{config.recovery.noise_0.weight_inject.use}_noise_{min_noise_intensity}_{max_noise_intensity}_{_min_noise_intensity}_{_max_noise_intensity}_{config.training.use_fl}'


save_dir = './save_model/' + basic_dir
model_path = config.data.architecture + '.pth'
save_path = os.path.join(save_dir, model_path)

# Training parameters
random_seed = 2024
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
    
early_stopping = EarlyStopping(patience=20, verbose=True)

        
def select_model(config,state='client', noise = None, is_train=None):
    if config.data.architecture == 'vgg8':
        if config.data.dataset == 'mnist':
            model = vgg.vgg8(in_channels=1, num_classes=10, noise_backbone=noise, is_train=is_train)
        elif config.data.dataset == 'cifar10':
            model = vgg.vgg8(in_channels=3, num_classes=10, noise_backbone=noise, is_train=is_train)
    elif config.data.architecture == 'resnet18':
        if config.data.dataset == 'mnist':
            model = resnet.resnet18(in_channels=1, noise_backbone=noise, is_train = is_train)
        elif config.data.dataset == 'cifar10':
            model =  resnet.resnet18(in_channels=3, noise_backbone=noise, is_train = is_train)
    elif config.data.architecture == 'lenet':
        if config.data.dataset == 'mnist':
            model = lenet.LeNet(in_channels=1, is_train = is_train, noise_backbone = noise)
        elif config.data.dataset == 'cifar10':
            model = lenet.LeNet(in_channels=3, is_train = is_train, noise_backbone = noise) 
    elif config.data.architecture == 'mlp':
        if config.data.dataset == 'mnist':
            model = mlp.MLPM(is_train = is_train,noise_backbone=noise)
        elif config.data.dataset == 'cifar10':
            model = mlp.MLPM(is_train = is_train, noise_backbone=noise)
    return model

def main():
    args.use_fl = config.training.use_fl
    args.client_num_in_total = config.training.client_num_in_total
    config.use_wandb = args.use_wandb
    """Train a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(save_dir, exist_ok=True)

    # step1: Load datasets.
    dataset = load_dataset(data_dir, 
                           config.training.batch_size, 
                           config.data.dataset)
    train_data, validation_data = dataset.load_images(config=config)

    # step2: Load Model Trainer
    model_trainer = FedProxTrainer()

    # step3: Model Definition
    """part1: global model"""
    global_model = select_model(config=config,state='global',noise=0, is_train=False)
    if args.use_foundation_model:
        global_model.load_state_dict(torch.load('../save_model/vgg11_cifar10_client_1_epoch_5_False_False_noise_0.1_0.5_0.1_0.1_False/client_0/vgg11_client_0_round_11_90.050000.pth.tar'))
    global_model.to(device)

    w_global = model_trainer.get_model_params(global_model)

    """part2: client model"""
    local_models = []
    for _ in range(args.client_num_in_total):
        if _ == 0:
            config.recovery.noise = config.recovery.noise_0
        elif _ == 1:
            config.recovery.noise = config.recovery.noise_1
        elif _ == 2:
            config.recovery.noise = config.recovery.noise_2
        elif _ == 3:
            config.recovery.noise = config.recovery.noise_3
        elif _ == 4:
            config.recovery.noise = config.recovery.noise_4

        model = select_model(config=config,noise=config.recovery.noise.act_inject.sigma)

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model_trainer.set_model_params(model,w_global)
        model.to(device)
        local_models.append(model)
    if args.use_wandb:
        wandb.init(project="AnalogAI", config=config)
        wandb.run.name  = basic_dir
    
    config.save_dir = save_dir
    fedproxAPI = FedProxAPI_personal(train_data=train_data,
                                     validation_data=validation_data,
                                     device=device,
                                     args=args,
                                     config=config,
                                     model_trainer=model_trainer,
                                     global_model=global_model,
                                     local_models=local_models
                                     )
    fedproxAPI.train()


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
