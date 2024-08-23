# -*- coding: utf-8 -*-
"""
created on 4.22
version: 1.0
train.py
"""


# Imports
import os
import csv
from altair import param, value
os.environ['WANDB_API_KEY'] = 'cfb5ba8f1bb02b39b518c24874b8579617459db3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from tqdm import tqdm
import re
import pandas as pd
# import timm
import numpy as np
# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, save
from torch import max as torch_max
from transformers import ViTModel
# Imports from networks.

from model.model_set import resnet, vgg, lenet, mobileNetv2, preact_resnet, vit
# Imports from utils.
from utils import create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from dataset import load_dataset
# from AnalogSram.sram_op import convert_to_sram_prepare
from AnalogSram.convert_sram import convert_to_sram_prepare

from args import parse_option
from utils import get_foundation_model,CustomViTModel
from transformers import ViTModel,ViTConfig


# from call_inference import infer_memtorch, infer_aihwkit, infer_MNSIM
# from call_inference import infer_aihwkit, infer_MemTorch
from call_inference import infer_aihwkit
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

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

# Device to use
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = config.training.device
# Path to store datasets
data_dir = os.path.join(os.getcwd(), "data", config.data.dataset)

# Path to store results

"""Normalize the experiment setup"""
min_noise_intensity = config.recovery.noise_0.act_inject.sigma
_min_noise_intensity = config.recovery.noise_0.weight_inject.level
if config.training.client_num_in_total == 5 or config.training.client_num_in_total == 1:
    max_noise_intensity = config.recovery.noise_4.act_inject.sigma
    _max_noise_intensity = config.recovery.noise_4.weight_inject.level
elif config.training.client_num_in_total == 10:
    max_noise_intensity = config.recovery.noise_9.act_inject.sigma
    _max_noise_intensity = config.recovery.noise_9.weight_inject.level

EXP_BASIC = True # TODO:是否进行基础实验

if EXP_BASIC:
    basic_dir = (args.config).split('.')[0]
    # save_dir = os.path.join('./save_model/',basic_dir,'client_2_0.1_0.1')
    save_dir = os.path.join('./save_model/',basic_dir)
    result_dict = {
        'Name': args.config,
        'Model': config.data.architecture,
        'Dataset': config.data.dataset,
    }
else:
    basic_dir = f'{config.data.architecture}_{config.data.dataset}_client_{config.training.client_num_in_total}_epoch_{config.training.epochs}_{config.recovery.noise_0.act_inject.use}_{config.recovery.noise_0.weight_inject.use}_noise_{min_noise_intensity}_{max_noise_intensity}_{_min_noise_intensity}_{_max_noise_intensity}_{config.training.use_fl}'
    save_path = os.path.join('./save_model/',basic_dir)
    client_name = 'client_0_0.0_0.1'
    if (args.config).split('_')[-1] == 'T.yml':
        save_dir = save_path
        current_act_intensity = None
        current_weight_intensity = None
    else:
        save_dir = save_path
        # save_dir = os.path.join(save_path, client_name) # TODO: 修改client inference
        current_act_intensity = float(client_name.split('_')[-2])
        current_weight_intensity = float(client_name.split('_')[-1])
    result_dict = {
        'Model': config.data.architecture,
        'Dataset': config.data.dataset,
        'Use FL':config.training.use_fl,
        'Cient Num': config.training.client_num_in_total,
        'Act Inject': config.recovery.noise_0.act_inject.use,
        'Weight Inject': config.recovery.noise_0.weight_inject.use,
        'Min Act Intensity': min_noise_intensity,
        'Max Act Intensity': max_noise_intensity,
        'Min Weight Intensity': _min_noise_intensity,
        'Max Weight Intensity': _max_noise_intensity,
        'Current Act Intensity': current_act_intensity,
        'Current Weight Intensity': current_weight_intensity,
        'Trick':'fused model:client_0_0_0.1'
    }

# Training parameters
random_seed = 2024
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)


def test_evaluation(validation_data, model, criterion, device):
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
    # for images, labels in validation_data:
    t = tqdm(validation_data, leave=False, total=len(validation_data))
    for _, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)
        # images = images
        # labels = labels
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

def get_best_model(save_dir):
    pattern = r"(\d+\.\d+)"
    model_list = os.listdir(save_dir)
    best_acc = 0.0
    best_model = None
    for model in model_list:
        match = re.search(pattern, model)
        if match:
            acc = float(match.group(0))
            if acc>best_acc and acc<100:
                best_acc = acc
                best_model = model
    return best_model

def select_model(config,state='client'):
    if config.data.architecture == 'vgg11':
        if config.data.dataset == 'mnist':
            model = vgg.VGG('VGG11',in_channels=1)
        elif config.data.dataset == 'cifar10':
            model = vgg.VGG('VGG11',in_channels=3)
    elif config.data.architecture == 'vgg16':
        if config.data.dataset == 'mnist':
            model = vgg.VGG('VGG16',in_channels=1)
        elif config.data.dataset == 'cifar10':
            model = vgg.VGG('VGG16',in_channels=3)
    elif config.data.architecture == 'resnet18':
        if config.data.dataset == 'mnist':
            model = resnet.resnet18(in_channels=1)
        elif config.data.dataset == 'cifar10':
            model = resnet.resnet18(in_channels=3)
    elif config.data.architecture == 'mobileNet':
        if config.data.dataset == 'mnist':
            model = mobileNetv2.MobileNetV2(in_channels=1)
        elif config.data.dataset == 'cifar10':
            model = mobileNetv2.MobileNetV2(in_channels=3)        
    elif config.data.architecture == 'vit':
        if config.data.dataset == 'mnist':
            _in_channels = 1
        elif config.data.dataset == 'cifar10':
            _in_channels = 3
        vit_config = ViTConfig(
            num_channels= _in_channels,
            image_size=32
        )
        model = CustomViTModel(vit_config=vit_config)
        # model = vit.VisionTransformer(img_size=32,
        #                             in_chans=_in_channels,
        #                             num_classes=10,
        #                             use_return_dict=False)
    
    if (args.config).split('_')[-1] != 'baseline.yml' and state == 'global':
        foundation_model_path = get_foundation_model(config=config)
        model.load_state_dict(torch.load(foundation_model_path))
    return model



def main():
    """Infer a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)

    # Load datasets.
    dataset = load_dataset(data_dir, config.training.batch_size, config.data.dataset)
    train_data, validation_data = dataset.load_images(config=config)

    #----Load the pytorch model------
    model = select_model(config=config)
    # if config.data.architecture == 'vgg11':
    #     model = vgg.VGG('VGG11')
    # elif config.data.architecture == 'vgg16':
    #     model = vgg.VGG('VGG16')
    # elif config.data.architecture == 'resnet18':
    #     model = resnet.ResNet18()
    # elif config.data.architecture == 'vit':
    #     model = ViTModel()

    model.to(device)   
    criterion = nn.CrossEntropyLoss().cuda()

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model) 

    """(fix): layer name transfer"""
    # trained_model = torch.load('/code/AnalogAI/save_model/deprecated/resnet_139_91.880000.pth.tar')
    # converted_weights = {}
    # for key, value in trained_model.items():
    #     if key.startswith('module.'):
    #         new_key = key[len('module.'):]
    #         converted_weights[new_key] = value
    #     else:
    #         converted_weights[key] = value
    # # 加载转换后的权重到模型
    # model.load_state_dict(converted_weights)

    #----load existing model---------
    print('save_dir:', save_dir)
    best_model = get_best_model(save_dir=save_dir)
    print('best_model:', best_model)
    if os.path.exists(os.path.join(save_dir, best_model)):
        print('==> loading existing model')
        model.load_state_dict(torch.load(os.path.join(save_dir, best_model)))   
    # model.load_state_dict(torch.load('/root/jiaqiLv/AnalogAI/save_model/done/resnet18_cifar10_client_5_epoch_1_True_False_noise_0.0_0.2_0.1_0.1_False/client_0_0_0.1/resnet18_client_0_round_10_86.120000.pth.tar'))

    """(test): aggregate"""
    # submodel_folder = os.listdir(save_dir)
    # submodel_list = []
    # for submodel_name in submodel_folder:
    #     _save_dir = os.path.join(save_dir,submodel_name)
    #     best_model = get_best_model(save_dir=_save_dir)
    #     if os.path.exists(os.path.join(_save_dir, best_model)):
    #         print('==> loading existing model')
    #         submodel_list.append(torch.load(os.path.join(_save_dir, best_model)))
    # model = resnet.ResNet18()
    # model = model.to(device=device)
    # model_params = {}
    # for submodel in submodel_list:
    #     for name,param in submodel.items():
    #         if name not in model_params:
    #             model_params[name] = 0.2*param
    #         else:
    #             model_params[name] += 0.2*param
    # model.load_state_dict(model_params)
    """ [test end] """

    """(optional): data parallel"""
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    
    #----------------------
    # wandb.login()
    # wandb.init(project="AnalogAI", config=config)
    # basic_name = (args.config).replace('.yml','') 
    # wandb.run.name = f'{basic_name}_inference'

    # infer
    if config.inference.platform.sram.use:
        print("==> inferencing on SRAM") 
        ps = [16, 32, 64, 128]
        es = np.linspace(0, 0.05, num=6, endpoint=True)
        for p in ps:
            for e in es: 
                infer_model_sram = convert_to_sram_prepare(model=model, 
                                                      device=device,
                                                      backend='SRAM', 
                                                      parallelism=int(p),
                                                      error=int(e),)
                _, _, error, accuracy = test_evaluation(
                                validation_data, infer_model_sram, criterion, device
                            )
                result_dict[f'SRAM_{p}_{e}'] = accuracy
                print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' parallelism:{p:.4f}' + f' error:{e:.4f}')
                # wandb.log({'parallelis_sram':p, 'error_sram':e, 'accuracy_sram':accuracy})

    if config.inference.platform.aihwkit.use:
        print("==> inferencing on IBM") 
        w = np.linspace(0., 0.2, num=10, endpoint=True)
        for n_w in w:
            infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(model)
            _, _, error, accuracy = test_evaluation(
                            validation_data, infer_model_aihwkit, criterion, device
                        )
            result_dict[f'AIHWKIT_{n_w}'] = accuracy
            print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')
            # wandb.log({'w_noise_aihwkit':n_w, 'error_aihwkit':error, 'accuracy_aihwkit':accuracy})
    
    result_file_path = f'./result/{config.data.dataset}_{config.data.architecture}_result_basic.csv'
    is_file_exists = os.path.isfile(result_file_path)
    with open(result_file_path,'a',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=result_dict.keys())
        if not is_file_exists:
            writer.writeheader()
    with open(result_file_path,'a',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=result_dict.keys())
        writer.writerow(result_dict)

    if config.inference.platform.memtorch.use:
        """
        unfinished
        """
        print("==> inferencing on MemTorch") 
        w = np.linspace(0., 0.02, num=5, endpoint=True)
        for n_w in w:
            # infer_model = infer_MemTorch(forward_w_noise=n_w).patch(model)
            # _, _, error, accuracy = test_evaluation(
            #                 validation_data, infer_model, criterion
            #             )
            print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')
    
    # wandb.finish()

if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()