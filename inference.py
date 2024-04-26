# -*- coding: utf-8 -*-
"""
created on 4.22
version: 1.0
train.py
"""


# Imports
import os

from altair import value
os.environ['WANDB_API_KEY'] = 'cfb5ba8f1bb02b39b518c24874b8579617459db3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from tqdm import tqdm
import re
# import timm
import numpy as np
# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, save
from torch import max as torch_max
# Imports from networks.

from model.model_set import resnet, vgg, lenet, mobileNetv2, preact_resnet, vit
# Imports from utils.
from utils import create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from dataset import load_dataset
# from AnalogSram.sram_op import convert_to_sram_prepare
from AnalogSram.convert_sram import convert_to_sram_prepare

from args import parse_option


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

basic_dir = f'{config.data.architecture}_{config.data.dataset}_client_{config.training.client_num_in_total}_epoch_{config.training.epochs}_{config.recovery.noise_0.act_inject.use}_{config.recovery.noise_0.weight_inject.use}_noise_{min_noise_intensity}_{max_noise_intensity}_{_min_noise_intensity}_{_max_noise_intensity}_{config.training.use_fl}'
# save_dir = os.path.join('./save_model/',basic_dir,'client_4')
save_dir = os.path.join('./save_model/',basic_dir)

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
            if acc>best_acc:
                best_acc = acc
                best_model = model
    return best_model


def main():
    """Infer a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)

    # Load datasets.
    dataset = load_dataset(data_dir, config.training.batch_size, config.data.dataset)
    train_data, validation_data = dataset.load_images()

    #----Load the pytorch model------
    if config.data.architecture == 'vgg11':
        model = vgg.VGG('VGG11')
    elif config.data.architecture == 'vgg16':
        model = vgg.VGG('VGG16')
    elif config.data.architecture == 'resnet18':
        model = resnet.ResNet18()
    elif config.data.architecture == 'vit':
        model = vit.ViT()

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
    model.load_state_dict(torch.load('/root/jiaqiLv/AnalogAI/save_model/vgg16_cifar10_client_5_epoch_1_True_False_noise_0.0_0.2_0.1_0.1_False/client_2/vgg16_client_2_round_10_80.650000.pth.tar'))

    # best_model = get_best_model(save_dir=save_dir)
    # if os.path.exists(os.path.join(save_dir, best_model)):
    #     print('==> loading existing model')
    #     model.load_state_dict(torch.load(os.path.join(save_dir, best_model)))   

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
        es = np.linspace(100, 200, num=2, endpoint=True)
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
                print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' parallelism:{p:.4f}' + f' error:{e:.4f}')
                # wandb.log({'parallelis_sram':p, 'error_sram':e, 'accuracy_sram':accuracy})

    if config.inference.platform.aihwkit.use:
        print("==> inferencing on IBM") 
        w = np.linspace(0., 0.1, num=20, endpoint=True)
        for n_w in w:
            infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(model)
            _, _, error, accuracy = test_evaluation(
                            validation_data, infer_model_aihwkit, criterion, device
                        )
            print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')
            # wandb.log({'w_noise_aihwkit':n_w, 'error_aihwkit':error, 'accuracy_aihwkit':accuracy})

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