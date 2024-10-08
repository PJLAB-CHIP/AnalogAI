# -*- coding: utf-8 -*-
"""
created on 10.26
version: 1.0
utils.py
"""
# Imports from PyTorch.
import torch
from torch import nn, Tensor
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from datetime import datetime
from noise_inject import InjectForward, InjectWeight, InjectWeightNoise
import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def create_optimizer(model, lr, momentum, weight_decay, optim_select, use_sam, ASAM):
    """Create the optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate
        optim_select (bool): optimizer to select

    Returns:
        Optimizer: created  optimizer
    """
    if optim_select == "Adam":
        optimizer = torch.optim.Adam
        print('use adam')
    else:
        optimizer = torch.optim.SGD
        print('use SGD')

    if use_sam:
        optimizer = SAM(model.parameters(), optimizer, adaptive=ASAM, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optimizer(model.parameters(), lr=lr)
    return optimizer

    
    
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FGSMTrainer:
    def __init__(self, model, loss_fn, optimizer, epsilon=0.03):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epsilon = epsilon

    def fgsm_attack(self, images, labels):
        # Set the model to evaluation mode
        self.model.eval()

        # Enable gradient calculation for the input images
        images.requires_grad = True

        # Forward pass
        outputs = self.model(images)
        
        # Calculate the loss
        loss = self.loss_fn(outputs, labels)
        
        # Zero gradients
        self.model.zero_grad()

        # Backward pass to calculate gradients
        loss.backward()

        # Collect the element-wise sign of the data gradient
        data_grad = images.grad.data.sign()

        # Create perturbed image by adjusting each pixel of the input image
        perturbed_images = images + self.epsilon * data_grad

        # Clip perturbed image values to be within the valid range [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return perturbed_images

    def train_step(self, inputs, labels):
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        # Zero gradients, backward pass, and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Generate adversarial examples using FGSM
        perturbed_inputs = self.fgsm_attack(inputs, labels)

        # Forward pass on adversarial examples
        adv_outputs = self.model(perturbed_inputs)
        adv_loss = self.loss_fn(adv_outputs, labels)

        # Total loss including adversarial examples
        total_loss = loss + adv_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class PGDTrainer:
    def __init__(self, model, loss_fn, optimizer, epsilon=0.03, num_steps=10, step_size=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def pgd_attack(self, images, labels):
        # Set the model to evaluation mode
        self.model.eval()

        # Enable gradient calculation for the input images
        images.requires_grad = True

        # Initialize perturbed image with original image
        perturbed_images = images.clone()

        for _ in range(self.num_steps):
            # Forward pass
            outputs = self.model(perturbed_images)

            # Calculate the loss
            loss = self.loss_fn(outputs, labels)

            # Zero gradients
            self.model.zero_grad()

            # Backward pass to calculate gradients
            loss.backward()

            # Collect the element-wise sign of the data gradient
            data_grad = perturbed_images.grad.data.sign()

            # Update perturbed image with small step in the direction of the gradient
            perturbed_images = perturbed_images + self.step_size * data_grad

            # Clip perturbed image values to be within the valid range [original_image - epsilon, original_image + epsilon]
            perturbed_images = torch.max(torch.min(perturbed_images, images + self.epsilon), images - self.epsilon)

        return perturbed_images.detach()

    def train_step(self, inputs, labels):
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        # Zero gradients, backward pass, and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Generate adversarial examples using PGD
        perturbed_inputs = self.pgd_attack(inputs, labels)

        # Forward pass on adversarial examples
        adv_outputs = self.model(perturbed_inputs)
        adv_loss = self.loss_fn(adv_outputs, labels)

        # Total loss including adversarial examples
        total_loss = loss + adv_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class Trainer:
    def __init__(self, model, loss_fn, optimizer, epsilon=0.03, num_steps=10, step_size=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def fgsm_attack(self, images, labels):
        # Set the model to evaluation mode
        self.model.eval()

        # Enable gradient calculation for the input images
        images.requires_grad = True

        # Forward pass
        outputs = self.model(images)
        
        # Calculate the loss
        loss = self.loss_fn(outputs, labels)
        
        # Zero gradients
        self.model.zero_grad()

        # Backward pass to calculate gradients
        loss.backward()

        # Collect the element-wise sign of the data gradient
        data_grad = images.grad.data.sign()

        # Create perturbed image by adjusting each pixel of the input image
        perturbed_images = images + self.epsilon * data_grad

        # Clip perturbed image values to be within the valid range [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return perturbed_images
    
    def pgd_attack(self, images, labels):
        # Set the model to evaluation mode
        self.model.eval()

        # Enable gradient calculation for the input images
        images.requires_grad = True

        # Initialize perturbed image with original image
        perturbed_images = images.clone()

        for _ in range(self.num_steps):
            # Forward pass
            outputs = self.model(perturbed_images)

            # Calculate the loss
            loss = self.loss_fn(outputs, labels)

            # Zero gradients
            self.model.zero_grad()

            # Backward pass to calculate gradients
            loss.backward()

            # Collect the element-wise sign of the data gradient
            data_grad = perturbed_images.grad.data.sign()

            # Update perturbed image with small step in the direction of the gradient
            perturbed_images = perturbed_images + self.step_size * data_grad

            # Clip perturbed image values to be within the valid range [original_image - epsilon, original_image + epsilon]
            perturbed_images = torch.max(torch.min(perturbed_images, images + self.epsilon), images - self.epsilon)

        return perturbed_images.detach()

    def train_step(self, inputs, labels):
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        # Zero gradients, backward pass, and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Generate adversarial examples using FGSM
        perturbed_inputs = self.fgsm_attack(inputs, labels)

        # Forward pass on adversarial examples
        adv_outputs = self.model(perturbed_inputs)
        adv_loss = self.loss_fn(adv_outputs, labels)

        # Total loss including adversarial examples
        total_loss = loss + adv_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    

def train_step(train_data, 
               model, 
               criterion, 
               optimizer, 
               device, 
               noise_a=None, 
               noise_w=None):

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

    if noise_a is not None:
        noise_a(model)
    total_loss = 0

    criterion.to(device)
    t = tqdm(train_data, leave=False, total=len(train_data))
    # for images, labels in train_data:
    for _, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        model.train()

        if isinstance(optimizer, SAM):
            # first forward-backward pass
            output = model(images)
            loss = criterion(output, labels)  # use this loss for any training statistics
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # second forward-backward pass
            criterion(model(images), labels).backward()  # make sure to do a full forward pass
            if isinstance(noise_w, InjectWeightNoise):
                noise_w.add_noise_to_weights()
                noise_w.update_model_weights()
            optimizer.second_step(zero_grad=True)
        else:
            # Add training Tensor to the model (input).
            output = model(images)
            loss = criterion(output, labels)

            # Run training (backward propagation).
            loss.backward()
            if isinstance(noise_w, InjectWeightNoise):
                noise_w.add_noise_to_weights()
                noise_w.update_model_weights()
            # Optimize weights.
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss

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
    criterion.to(device)
    # device_cuda = torch.device("cuda:6")
    # model.to(device)

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

def theory_driven_loss(outputs, targets, weights, lambda_val, mu, eta):
    """
    implementation of 'Formalizing Generalization and Adversarial
      Robustness of Neural Networks to Weight Perturbations'
    eg.
    # outputs = model(data)
    # weights = [param for param in model.parameters() if len(param.size()) > 1]  # Assuming weights are 2D
    # eta = compute_eta(outputs, target)  # This function needs to be defined based on your robustness analysis
    # loss = theory_driven_loss(outputs, target, weights, lambda_val=0.01, mu=0.01, eta=eta)

    :param outputs: Tensor, the predictions of the network (logits before softmax)
    :param targets: Tensor, the true labels
    :param weights: List of Tensors, the weight matrices of each layer of the network
    :param lambda_val: float, coefficient for the robustness loss
    :param mu: float, coefficient for the regularization term
    :param eta: Tensor, precomputed robustness loss terms for each class
    :return: computed loss value
    """
    cls_loss = F.cross_entropy(outputs, targets)

    robustness_loss = torch.max(eta.gather(1, targets.unsqueeze(1)), dim=1)[0]

    reg_loss = sum(torch.norm(w, p=1, dim=1).max() + torch.norm(w.t(), p=1, dim=1).max() for w in weights)

    total_loss = cls_loss + lambda_val * robustness_loss + mu * reg_loss

    return total_loss

