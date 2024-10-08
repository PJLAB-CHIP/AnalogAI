# -*- coding: utf-8 -*-
"""
created on 4.22
version: 1.0
train.py
"""


# Imports
import os
# os.environ['WANDB_API_KEY'] = 'e7a84490fccf4d551013cad7ca58549bb09594f7'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from datetime import datetime
from tqdm import tqdm
# import timm
import timm
import numpy as np
# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, save
from torch import max as torch_max
import torch.nn.functional as F
from torch.optim import lr_scheduler
# Imports from networks.

from model import resnet, vgg, lenet, mobileNetv2, preact_resnet, vit
# Imports from utils.
from utils import create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from dataset import load_dataset
# from AnalogSram.sram_op import convert_to_sram_prepare
from AnalogSram.convert_sram import convert_to_sram_prepare


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

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='AnalogAI', type=str, help='name')
parser.add_argument('--architecture', default='vit', type=str, help='name')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size')
parser.add_argument('-sram_analog_recover',default=False, type=bool, help='whether to recover on an analog platform')
parser.add_argument('-analog_infer',default=False, type=bool, help='whether to infer on an analog platform')

parser.add_argument('--config', type=str, default='exp1_infer.yml', help='Path to the config file')
args = parser.parse_args()

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

# Device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_cpu = torch.device("cpu")

# Path to store datasets
data_dir = os.path.join(os.getcwd(), "data", config.data.dataset)

# Path to store results
save_dir = './save_model/' + config.data.architecture
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


def main():
    """Infer a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)

    # Load datasets.
    dataset = load_dataset(data_dir, args.batch_size, args.dataset)
    train_data, validation_data = dataset.load_images()

    #----Load the pytorch model------

    model = resnet.ResNet18()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)     
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()

    #----load existing model---------
    if os.path.exists(os.path.join(save_dir, 'checkpoint_.pth.tar')):
        print('==> loading existing model')
        model.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoint_.pth.tar')))    
    #----------------------
    # wandb.login()
    # wandb.init(project="AnalogAI", config=config)

    # infer
    if config.inferemce.platform.sram:
        print("==> inferencing on SRAM") 
        ps = [16, 32, 64, 128]
        es = np.linspace(0, 300, num=4, endpoint=True)
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
                # wandb.log({'parallelis':p, 'error':e, 'accuracy':accuracy})

    if config.inferemce.platform.aihwkit:
        print("==> inferencing on IBM") 
        w = np.linspace(0., 0.02, num=5, endpoint=True)
        for n_w in w:
            infer_model_aihwkit = infer_aihwkit(forward_w_noise=n_w).patch(model)
            _, _, error, accuracy = test_evaluation(
                            validation_data, infer_model_aihwkit, criterion, device_cpu
                        )
            print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')

    if config.inferemce.platform.memtorch:
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