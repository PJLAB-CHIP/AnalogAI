
# Imports from PyTorch.
import torch
from torch import nn, Tensor
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

def create_optimizer(model, learning_rate, optim_select):
    """Create the optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate
        optim_select (bool): optimizer to select

    Returns:
        Optimizer: created  optimizer
    """
    if optim_select:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print('use adam')
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print('use SGD')
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
    
