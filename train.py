# -*- coding: utf-8 -*-
"""
created on 10.26
version: 1.0
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime
from tqdm import tqdm
import timm
import numpy as np
# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, save
from torch import max as torch_max
import torch.nn.functional as F
# Imports from networks.
from network import create_resnet32_model
from model import resnet_Q, resnet, vgg, vggQ, lenet, lenetQ, mobileNetv2, mobileNetv2Q, preact_resnet
# Imports from utils.
from utils import create_optimizer, SAM, FGSMTrainer, PGDTrainer
from dataset import load_dataset
# Imports from networks.
from noise_inject import InjectForward, InjectWeight, InjectWeightNoise

# from call_inference import infer_memtorch, infer_aihwkit, infer_MNSIM
from call_inference import infer_aihwkit, infer_MNSIM
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size')
parser.add_argument('-lr',default=1e-1, type=float, help='learning rate')
parser.add_argument('-momentum',default=9e-1, type=float, help='momentum')
parser.add_argument('-weight_decay',default=0.0005, type=float, help='weight_decay')
parser.add_argument('-epochs', default=250, type=int, help='sum of epochs')
parser.add_argument('-n_classes', default=10, type=int, help='number of categories')
parser.add_argument('--use_sgd', help='whether use sgd or adam', action='store_true')
parser.add_argument('--injectforward', default=False, type=bool, help='whether inject forward noise or not')
parser.add_argument('--injectweightnoise', default=True, type=bool, help='whether inject weight noise or not')
parser.add_argument('--injectweight', default=False, type=bool, help='whether inject noise or not')
parser.add_argument('--mean', default=1., type=float, help='average value of noise')
parser.add_argument('--sigma', default=1.0 / 6.0, type=float, help='variance of the noise')
parser.add_argument('--SAM', default=False, type=bool, help='whether use SAM')
parser.add_argument('-adaptive',default=False, type=bool, help='whether use ASAM')
args = parser.parse_args()
# Device to use
USE_CUDA = 0
if torch.cuda.is_available():
    USE_CUDA = 1
device = torch.device("cuda:6" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
save_dir = './save_model/' + 'RESNET'
model_path = 'res50_jw.pth'
WEIGHT_PATH = os.path.join(save_dir, model_path)

# Training parameters
random_seed = 1
# N_EPOCHS = 30
N_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.1
N_CLASSES = 10
mask = None                                                    



def train_step(train_data, model, criterion, optimizer, noise=None):
    """Train network.

    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        noise: noise type
    Returns:
        nn.Module, Optimizer, float: model, optimizer, and epoch loss
    """
    if isinstance(noise, InjectForward):
        noise(model)
    total_loss = 0
    device_cuda = torch.device("cuda:6")
    model.to(device_cuda)
    model.train()
    t = tqdm(train_data, leave=False, total=len(train_data))
    # for images, labels in train_data:
    for _, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if isinstance(optimizer, SAM):
            # first forward-backward pass
            output = model(images)
            loss = criterion(output, labels)  # use this loss for any training statistics
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # second forward-backward pass
            criterion(model(images), labels).backward()  # make sure to do a full forward pass
            if isinstance(noise, InjectWeightNoise):
                noise.add_noise_to_weights()
                noise.update_model_weights()
            optimizer.second_step(zero_grad=True)
        else:
            # Add training Tensor to the model (input).
            output = model(images)
            loss = criterion(output, labels)

            # Run training (backward propagation).
            loss.backward()
            if isinstance(noise, InjectWeightNoise):
                noise.add_noise_to_weights()
                noise.update_model_weights()
            # Optimize weights.
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss


def test_evaluation(validation_data, model, criterion):
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
    device_cuda = torch.device("cuda:6")
    model.to(device_cuda)

    # for images, labels in validation_data:
    t = tqdm(validation_data, leave=False, total=len(validation_data))
    for _, (images, labels) in enumerate(t):
        images = images.to(device_cuda)
        labels = labels.to(device_cuda)
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

def training_loop(model, criterion, optimizer, train_data, validation_data, epochs, print_every=1):
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
        if epoch==1 and args.injectforward:
            print('====>inject forward noise<====')
            noise = InjectForward('normal_flu', args.mean, args.sigma, mask)
            # noise(model)
        elif epoch==1 and args.injectweightnoise:
            print('====>inject weight noise<====')
            noise = InjectWeightNoise(model, noise_level=0.1)
        else:
            noise = None
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer, noise)
        train_losses.append(train_loss)

        if epoch % print_every == (print_every - 1):
            # Validate_step
            with no_grad():
                model, valid_loss, error, accuracy = test_evaluation(
                    validation_data, model, criterion
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

    return model, optimizer


def main():
    """Train a PyTorch CNN analog model with dataset (eg. CIFAR10)."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)

    # Load datasets.
    dataset = load_dataset(PATH_DATASET, args.batch_size, args.dataset)
    train_data, validation_data = dataset.load_images()

    #---------------------- Load the pytorch model
    # model = create_resnet32_model(args.n_classes)
    # model = resnet.ResNet18()
    # model = vgg.VGG('VGG16')
    # model = mobileNetv2.MobileNetV2()
    # model = preact_resnet.PreActResNet18()
    model = resnet.ResNet50()
    # model = resnet_Q.ResNet18Q()
    # model = resnet.ResNet18()
    # model_name = "vit_base_patch16_224"  # 选择 VIT 模型的名称
    # model = timm.create_model(model_name, num_classes=10, in_chans=3, pretrained=True)
    # print('create vit')
    #---------------------------------------------------
    if os.path.exists(os.path.join(save_dir, 'res50_jw.pth')):
        # load existing model
        print('==> loading existing model')
        model.load_state_dict(torch.load(os.path.join(save_dir, 'res50_jw.pth')))    
    # model = timm.create_model('resnet50', pretrained=False, num_classes=10)
    #----------------------
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)     
    model.to(device)

    base_optimizer = create_optimizer(model, args.lr, args.use_sgd)
    if args.SAM:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = base_optimizer
    criterion = nn.CrossEntropyLoss()

    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started Training")
    # if args.quantize:

    model, optimizer = training_loop(
        model, criterion, optimizer, train_data, validation_data, N_EPOCHS
    )

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed Network Training")

    torch.save(model.state_dict(), WEIGHT_PATH)
    print('==> saving trained model' + model_path)
    # try:
    #     model.load_state_dict(WEIGHT_PATH, torch.load('checkpoint.pth'), strict=True)
    #     model.eval()
    # except:
    #     raise Exception('trained_model.pt has not been found.')
    
    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Started DNN inferencing")
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
