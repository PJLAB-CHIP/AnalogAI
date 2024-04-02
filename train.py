# -*- coding: utf-8 -*-
"""
created on 10.26
version: 1.0
train.py
"""
# pylint: disable=invalid-name

# Imports
import os
os.environ['WANDB_API_KEY'] = 'e7a84490fccf4d551013cad7ca58549bb09594f7'
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
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
import timm
from network import create_resnet32_model
from model import resnet_Q, resnet, vgg, vggQ, lenet, lenetQ, mobileNetv2, mobileNetv2Q, preact_resnet, vit
# Imports from utils.
from utils import create_optimizer, SAM, FGSMTrainer, PGDTrainer, train_step, test_evaluation
from dataset import load_dataset
# Imports from networks.
from noise_inject import InjectForward, InjectWeight, InjectWeightNoise
from earlystopping import EarlyStopping

from AnalogSram.sram_op import convert_to_sram_prepare


# from call_inference import infer_memtorch, infer_aihwkit, infer_MNSIM
from call_inference import infer_aihwkit, infer_MNSIM
import argparse
import wandb
wandb.login()
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='AnalogAI', type=str, help='name')
parser.add_argument('--architecture', default='vit', type=str, help='name')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size')
parser.add_argument('-lr',default=1e-3, type=float, help='learning rate')
parser.add_argument('-momentum',default=9e-1, type=float, help='momentum')
parser.add_argument('-weight_decay',default=0.0005, type=float, help='weight_decay')
parser.add_argument('-epochs', default=200, type=int, help='sum of epochs')
parser.add_argument('-n_classes', default=10, type=int, help='number of categories')
parser.add_argument('--use_sgd', help='whether use sgd or adam', action='store_true')
parser.add_argument('--injectforward', default=False, type=bool, help='whether inject forward noise or not')
parser.add_argument('--mask', default=True, type=bool, help='whether inject forward with masked layers')
parser.add_argument('--injectweightnoise', default=False, type=bool, help='whether inject weight noise or not')
parser.add_argument('--injectweight', default=False, type=bool, help='whether inject noise or not')
parser.add_argument('--mean', default=1., type=float, help='average value of noise')
parser.add_argument('--sigma', default=1.0 / 6.0, type=float, help='variance of the noise')
parser.add_argument('--SAM', default=False, type=bool, help='whether use SAM')
parser.add_argument('-adaptive',default=False, type=bool, help='whether use ASAM')
parser.add_argument('-analog_recover',default=True, type=bool, help='whether to recover on an analog platform')
parser.add_argument('-analog_infer',default=False, type=bool, help='whether to infer on an analog platform')
args = parser.parse_args()

config = vars(args)
print(config)

# Device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to store datasets
data_dir = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
save_dir = './save_model/' + 'RESNET'
model_path = 'res18.pth'
WEIGHT_PATH = os.path.join(save_dir, model_path)

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
                epochs, 
                mean,
                sigma,
                mask,
                injectforward, 
                injectweightnoise,
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

    # Train model
    for epoch in range(0, epochs):
        # Train_step
        if epoch==1 and injectforward:
            print('====>inject forward noise<====')
            noise = InjectForward('normal_flu', mean, sigma, mask)
            # noise(model)
        elif epoch==1 and injectweightnoise:
            print('====>inject weight noise<====')
            noise = InjectWeightNoise(model, noise_level=0.1)
        else:
            noise = None

        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer, device, noise)
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
        best_accuracy = early_stopping(accuracy, model.state_dict(), epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        wandb.log({'epoch':epoch, 'accuracy':best_accuracy, 'train_loss':train_loss, 'valid_loss':valid_loss})
    
    wandb.finish()
    
    return model, optimizer


def main():
    """Train a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)

    # Load datasets.
    dataset = load_dataset(data_dir, args.batch_size, args.dataset)
    train_data, validation_data = dataset.load_images()

    #----Load the pytorch model------
    # model = resnet_Q.ResNet18Qsram()
    # model = resnet.ResNet18()
    # model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=10)
    # model = mobileNetv2.MobileNetV2()
    model = vit.ViT()
    # model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)     
    model.to(device)

    if args.analog_recover:
    #----load existing model---------
        if os.path.exists('/code/AnalogAI/save_model/RESNET/vit_72_81.690000.pth.tar'):
            print('==> loading existing model')
            model.load_state_dict(torch.load('/code/AnalogAI/save_model/RESNET/vit_72_81.690000.pth.tar'))   
        #----------------------
        model = convert_to_sram_prepare(model=model, device=device, backend='SRAM', parallelism=16, error=300,)
    
    optimizer = create_optimizer(model, args.lr, args.momentum, args.weight_decay, args.use_sgd, args.SAM, args.adaptive)

    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=True)
        
    criterion = nn.CrossEntropyLoss()

    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started Training")

    wandb.init(project="AnalogAI", config=config)

    model, optimizer = training_loop(
        model, 
        criterion, 
        optimizer, 
        train_data, 
        validation_data, 
        args.epochs, 
        args.mean, 
        args.sigma, 
        args.mask,
        args.injectforward, 
        args.injectweightnoise, 
        pla_lr_scheduler,
        device,

    )

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed Network Training")

    if args.analog_infer:
        print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Starting DNN inference on a analog platform")
        if os.path.exists(os.path.join(save_dir, 'res50jw.pth')):
            # load existing model
            model.eval()
            device_cpu = torch.device("cpu")
            model.to(device_cpu)
            print('==> loading existing model')
            model.load_state_dict(torch.load(WEIGHT_PATH))
            # # infer_model = infer_memtorch().patch(model)
            # infer_model = infer_aihwkit().patch(model)
            # print('==> inferencing on IBM')
            # _, _, error, accuracy = infer_evaluation(
            #                 validation_data, infer_model, criterion
            #             )
            # print(f'error:{error:.2f}' + f'/n accuracy:{accuracy:.2f}')
            w = np.linspace(0., 0.02, num=5, endpoint=True)
            # infer_model = infer_memtorch().patch(model)
            for n_w in w:
                infer_model = infer_aihwkit(forward_w_noise=n_w).patch(model)
                _, _, error, accuracy = test_evaluation(
                                validation_data, infer_model, criterion
                            )
                print(f'error:{error:.2f}' + f'accuracy:{accuracy:.2f}' + f' w_noise:{n_w:.4f}')


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
