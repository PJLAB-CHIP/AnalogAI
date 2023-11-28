import torch
# Imports from PyTorch.
import torch
from torch import nn, Tensor
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

class load_dataset():
    def __init__(self, PATH_DATASET, BATCH_SIZE, dataname) -> None:
          self.path = PATH_DATASET
          self.BATCH_SIZE = BATCH_SIZE
          self.dataname = dataname
    def load_images(self):
        """Load images for train from torchvision datasets.

        Returns:
            Dataset, Dataset: train data and validation data"""
        if self.dataname == 'cifar10' :
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                trainset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform_train)
                testset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform_test)


        if self.dataname == 'mnist' :
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                trainset = torchvision.datasets.MNIST(
                    root='./data',train=True,download=True,transform=transform)
                testset = torchvision.datasets.MNIST(
                    root='./data',train=False,download=True,transform=transform)

        if self.dataname == 'svhn':
                transform = transforms.Compose([transforms.Resize((32,32)),
                    transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                trainset = torchvision.datasets.SVHN(
                    root='./data',split='train',download=True,transform=transform)
                testset = torchvision.datasets.SVHN(
                    root='./data',split='test',download=True,transform=transform)

        train_data = torch.utils.data.DataLoader(
                trainset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=1)
        validation_data = torch.utils.data.DataLoader(
                testset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=1)
        return train_data, validation_data