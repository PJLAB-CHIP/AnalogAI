# -*- coding: utf-8 -*-
"""
created on 4.19
version: 1.0
train.py
"""

# Imports
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
os.environ['WANDB_API_KEY'] = 'cfb5ba8f1bb02b39b518c24874b8579617459db3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# import timm
import numpy as np
# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, save
# Imports from networks.
# import timm
from model.model_set import resnet, vgg, lenet, mobileNetv2, preact_resnet, vit
# Imports from utils.
from utils import dict2namespace, create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from dataset import load_dataset
# Imports from networks.
from earlystopping import EarlyStopping
# FedProx
from model_trainer_fedprox import FedProxTrainer
from fedprox_api import FedProxAPI_personal
from args import parse_option

import argparse
import yaml
import wandb
wandb.login()

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

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

# Device to use
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device = args.device

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
# initialize the early_stopping object
    
early_stopping = EarlyStopping(patience=20, verbose=True)

def main():
    args.use_fl = config.training.use_fl
    args.client_num_in_total = config.training.client_num_in_total
    """Train a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(save_dir, exist_ok=True)

    # step1: Load datasets.
    dataset = load_dataset(data_dir, 
                           config.training.batch_size, 
                           config.data.dataset)
    train_data, validation_data = dataset.load_images()

    # step2: Load Model Trainer
    model_trainer = FedProxTrainer()

    # step3: Model Definition
    """part1: global model"""
    if config.data.architecture == 'vgg11':
        global_model = vgg.VGG('VGG11')
    elif config.data.architecture == 'vgg16':
        global_model = vgg.VGG('VGG16')
    elif config.data.architecture == 'resnet18':
        global_model = resnet.ResNet18()
    elif config.data.architecture == 'vit':
        global_model = vit.ViT()
    if args.use_foundation_model:
        global_model.load_state_dict(torch.load('/root/jiaqiLv/AnalogAI/save_model/vgg11_cifar10_client_1_epoch_5_False_False_noise_0.1_0.5_0.1_0.1_False/client_0/vgg11_client_0_round_11_90.050000.pth.tar'))
    global_model.to(device)
    # if torch.cuda.device_count() > 1:
    #     global_model = nn.DataParallel(global_model)
    w_global = model_trainer.get_model_params(global_model)

    """part2: client model"""
    local_models = []
    for _ in range(args.client_num_in_total):
        if config.data.architecture == 'vgg11':
            model = vgg.VGG('VGG11')
        elif config.data.architecture == 'vgg16':
            model = vgg.VGG('VGG16')
        elif config.data.architecture == 'resnet18':
            model = resnet.ResNet18()
        elif config.data.architecture == 'vit':
            model = vit.ViT()
        model_trainer.set_model_params(model,w_global)
        local_models.append(model)
    
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
