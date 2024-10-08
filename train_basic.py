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
# os.environ['WANDB_API_KEY'] = 'e7a84490fccf4d551013cad7ca58549bb09594f7'
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
from model import resnet, vgg, lenet
# Imports from utils.
from data.dataset import load_dataset
# Imports from networks.
from recovery.noise_aware.noise_inject import InjectForward, InjectWeight, InjectWeightNoise
from recovery.qat.fake_quantize import fake_quantize_prepare
from utils.utils import test_evaluation, train_step, create_optimizer
from utils.earlystopping import EarlyStopping

# from call_inference import infer_memtorch, infer_aihwkit, infer_MNSIM
# from call_inference import infer_aihwkit, infer_MNSIM
import argparse
import yaml
import wandb
# wandb.login()

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

parser.add_argument('--config', type=str, default='exp1.yml', help='Path to the config file')
args = parser.parse_args()

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

# Device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# initialize the early_stopping object
    
early_stopping = EarlyStopping(patience=20, verbose=True)

def infer_evaluation(validation_data, model, criterion):
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
    device_cpu = torch.device("cpu")
    model.to(device_cpu)

    # for images, labels in validation_data:
    t = tqdm(validation_data, leave=False, total=len(validation_data))
    for _, (images, labels) in enumerate(t):
        images = images.to(device_cpu)
        labels = labels.to(device_cpu)
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

    # Load datasets.
    dataset = load_dataset(data_dir, 
                           config.training.batch_size, 
                           config.data.dataset)
    train_data, validation_data = dataset.load_images()

    #----Load the pytorch model------
    model = resnet.ResNet18()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)     
    model.to(device)
    
    optimizer = create_optimizer(model, 
                                 config.training.lr, 
                                 config.training.momentum, 
                                 config.training.weight_decay, 
                                 config.training.optimizer, 
                                 config.recovery.optimizer.sam, 
                                 config.recovery.optimizer.adaptive)

    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=True)
        
    criterion = nn.CrossEntropyLoss()

    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started Training")

    # wandb.init(project="AnalogAI", config=config)

    model, optimizer = training_loop(model, 
                                     criterion, 
                                     optimizer, 
                                     train_data, 
                                     validation_data, 
                                     config,
                                     pla_lr_scheduler,
                                     device,

    )

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed Network Training")

    # if args.analog_infer:
    #     print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Starting DNN inference on a analog platform")
    #     if os.path.exists(os.path.join(save_dir, 'res50jw.pth')):
    #         # load existing model
    #         model.eval()
    #         device_cpu = torch.device("cpu")
    #         model.to(device_cpu)
    #         print('==> loading existing model')
    #         model.load_state_dict(torch.load(save_path))
    #         # # infer_model = infer_memtorch().patch(model)
    #         # infer_model = infer_aihwkit().patch(model)
    #         # print('==> inferencing on IBM')
    #         # _, _, error, accuracy = infer_evaluation(
    #         #                 validation_data, infer_model, criterion
    #         #             )
    #         # print(f'error:{error:.2f}' + f'/n accuracy:{accuracy:.2f}')
    #         w = np.linspace(0., 0.02, num=5, endpoint=True)
    #         # infer_model = infer_memtorch().patch(model)
    #         for n_w in w:
    #             infer_model = infer_aihwkit(forward_w_noise=n_w).patch(model)
    #             _, _, error, accuracy = test_evaluation(
    #                             validation_data, infer_model, criterion
    #                         )
    #             print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
