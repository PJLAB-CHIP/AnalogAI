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
from datetime import datetime
from tqdm import tqdm
# import timm
import numpy as np
# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, save
from torch import max as torch_max
import torch.nn.functional as F
from torch.optim import lr_scheduler
# Imports from networks.
# import timm
from model.model_set import resnet, vgg, lenet, mobileNetv2, preact_resnet, vit
# Imports from utils.
from utils import dict2namespace, create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from dataset import load_dataset
# Imports from networks.
from noise_inject import InjectForward, InjectWeight, InjectWeightNoise
from qat.fake_quantize import fake_quantize_prepare
from earlystopping import EarlyStopping

from AnalogSram.sram_op import convert_to_sram_prepare

# FedProx
from model_trainer_fedprox import FedProxTrainer
from fedprox_api import FedProxAPI_personal
from args import parse_option


# from call_inference import infer_memtorch, infer_aihwkit, infer_MNSIM
# from call_inference import infer_aihwkit, infer_MNSIM
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to store datasets
data_dir = os.path.join(os.getcwd(), "data", config.data.dataset)

# Path to store results
basic_dir = (args.config).replace('.yml','')
save_dir = './save_model/' + f'{config.data.architecture}_{basic_dir}'
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

def training_loop(model, 
                  criterion, 
                  optimizer, 
                  train_data, 
                  validation_data, 
                  config,
                  pla_lr_scheduler,
                  device, 
                  print_every=1):
    """Training loop.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress

    Returns:
        nn.Module, Optimizer, Tuple: model, optimizer, and a tuple of
            lists of train losses, validation losses, and test error
    """
    train_losses = []
    valid_losses = []
    test_error = []

    if config.recovery.qat.use:
        model = fake_quantize_prepare(model=model, 
                                      device=device, 
                                      a_bits=config.recovery.qat.a_bits, 
                                      w_bits=config.recovery.qat.w_bits, )

    # Train model
    for epoch in range(0, config.training.epochs):
        # Train_step
        if  config.recovery.noise.act_inject.use:
            print('====>inject forward noise<====')
            noise_a = InjectForward(config.recovery.noise.act_inject.type, 
                                    config.recovery.noise.act_inject.mean, 
                                    config.recovery.noise.act_inject.sigma, 
                                    config.recovery.noise.act_inject.mask)
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
                                                  criterion, 
                                                  optimizer, 
                                                  device, 
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
        best_accuracy = early_stopping(accuracy, 
                                       model.state_dict(), 
                                       config.data.architecture,
                                       epoch, 
                                       save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # wandb.log({'epoch':epoch, 'accuracy':best_accuracy, 'train_loss':train_loss, 'valid_loss':valid_loss})
    
    # wandb.finish()
    
    return model, optimizer


def main():
    """Train a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(save_dir, exist_ok=True)
    CLIENT_NUM = 5

    # step1: Load datasets.
    dataset = load_dataset(data_dir, 
                           config.training.batch_size, 
                           config.data.dataset)
    train_data, validation_data = dataset.load_images()

    # step2: Load Model Trainer
    model_trainer = FedProxTrainer()

    args.client_num_in_total = CLIENT_NUM
    # step3: Model Definition
    """part1: global model"""
    global_model = resnet.ResNet18()
    # if torch.cuda.device_count() > 1:
    #     global_model = nn.DataParallel(global_model)
    global_model.to(device)
    w_global = model_trainer.get_model_params(global_model)
    """part2: client model"""
    local_models = []
    for _ in range(CLIENT_NUM):
        model = resnet.ResNet18()
        model_trainer.set_model_params(model,w_global)
        local_models.append(model)
    
    wandb.init(project="AnalogAI", config=config)
    wandb.run.name  = (args.config).replace('.yml','')
    
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

    # if args.sram_analog_recover:
    # #----load existing model---------
    #     if os.path.exists('.pth.tar'):
    #         print('==> loading existing model')
    #         model.load_state_dict(torch.load('.pth.tar'))   
    #     #----------------------
    #     model = convert_to_sram_prepare(model=model, device=device, backend='SRAM', parallelism=16, error=300,)
    
    # optimizer = create_optimizer(model, 
    #                              config.training.lr, 
    #                              config.training.momentum, 
    #                              config.training.weight_decay, 
    #                              config.training.optimizer, 
    #                              config.recovery.optimizer.sam, 
    #                              config.recovery.optimizer.adaptive)

    # pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                   factor=0.5,
    #                                                   patience=10,
    #                                                   verbose=True)
        
    # criterion = nn.CrossEntropyLoss()

    # print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started Training")

    # # wandb.init(project="AnalogAI", config=config)

    # model, optimizer = training_loop(model, 
    #                                  criterion, 
    #                                  optimizer, 
    #                                  train_data, 
    #                                  validation_data, 
    #                                  config,
    #                                  pla_lr_scheduler,
    #                                  device,

    # )

    # print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed Network Training")


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
