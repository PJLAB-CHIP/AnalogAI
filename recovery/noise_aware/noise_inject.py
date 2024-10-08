# -*- coding: utf-8 -*-
"""
created on 10.26
version: 1.0
noise_inject.py
"""
from types import MethodType
import torch
import torch.nn as nn
import numpy as np
from functools import partial

def negative_fluction(output, sigma, mean, sigma_global, sigma_neg):
    # print(sigma, mean, sigma_global, sigma_neg)
    noise_client = torch.randn_like(output) * sigma + mean
    # Subtract larger Gaussian noise (sigma * 2 for example)
    noise_neg = torch.randn_like(output) * (sigma_global * sigma_neg) + mean
    output = output * noise_client - output * noise_neg
    return output

class InjectForward():
    def __init__(self, fault_type='Gaussian', arg1=0., arg2=0., sigma_global=0.25, sigma_dict=None, layer_mask=None) -> None:
        # self.model = model
        self.fault_type = fault_type
        self.layer_mask = layer_mask
        self.arg1 = arg1 
        self.arg2 = arg2
        self.sigma_global = sigma_global
        self.sigma_dict = sigma_dict
        self.hook_list = []
        self.sigma_neg = 1.0
        
    def default_hook( module, input, output):
        print('hook',module)
        print('input',input)
        print('output',output)
        return output
        
    # def get_layers(self, model):
    #     real_layers=[]
    #     layers = list(model.modules())
    #     for layer in layers:
    #         if len(list(layer.children()))==0 and len(list(layer.parameters())) :
    #                 real_layers.append(layer)
    #         if len(list(layer.children())) and len(list(list(layer.children())[0].parameters()))==0  and len(list(layer.parameters())):
    #                 real_layers.append(layer)
    #     return real_layers
    
    def get_layers(self, model):
        real_layers={}
        # print(list(model.named_modules()))
        # print('------------------------')
        # print(list(model.modules()))
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                real_layers[name] = layer
        return real_layers
    
    def expand_mask(self, model):          
        layers = self.get_layers(model)
        if(self.layer_mask is None):
            self.layer_mask = 1+np.zeros(len(layers.keys()), dtype='bool')
        new_mask = self.layer_mask
        return new_mask
    
    def get_hook(self, model, function=default_hook):
        # layer_counter, coefficients = assign_coefficients_to_layers(model)
        # print('@@@--->', layer_counter)
        # print('###--->', coefficients)
        mask = self.expand_mask(model)
        layers = self.get_layers(model)
        for l,m in zip(layers, mask):
            if m:
                a = l.register_forward_hook(function)
                self.hook_list.append(a)
        # for i, (layer, mask_value) in enumerate(zip(layers, mask)):
        #     if mask_value:
        #         # sigma = self.sigma_dict.get(i, 1.0)
        #         sigma_neg = 1.0 
        #         # if self.fault_type == 'Negative Feedback':
        #         #     hook_function = partial(function, sigma_global=self.sigma_global, sigma_neg=sigma_neg)
        #         # else:
        #         #     hook_function = function
        #         print(function.__name__)
        #         print("?????????", layers[layer].forward, hasattr(layers[layer], "_orig_forward"))
        #         # first inject forward, set init forward to layer forward
        #         if not hasattr(layers[layer], "_orig_forward"):
        #             layers[layer]._init_forward = layers[layer].forward
        #             print("@@@@@ --->  init forward")
        #         assert hasattr(layers[layer], "_init_forward")
                
        #         layers[layer]._orig_forward = layers[layer]._init_forward
        #         def _new_foward(m, x):
        #             result = m._orig_forward(x)
        #             negative_fluction(result, self.arg1, self.arg2, self.sigma_global, coefficients[layer])
        #             return result
        #         layers[layer].forward = MethodType(_new_foward, layers[layer])
        #         # hook = layer.register_forward_hook(function, with_kwargs={"mean": self.arg1, "sigma": self.arg2, 
        #         #                                                             "sigma_global": self.sigma_global, 
        #         #                                                             "sigma_neg" : sigma_neg})
        #         # hook = layer.register_forward_hook(lambda module, input, output: hook_function(module, input, output))
        #         # hook = layer.register_forward_hook(lambda module, input, output: function(module, input, output, self.sigma_global, self.sigma_neg))
        #         # self.hook_list.append(hook)

    def __call__(self, model):
        
        def normal_fluction(module, input, output):          ## Gaussian noise 
            mean, sigma  = self.arg1, self.arg2
            w = output
            normal = torch.randn(w.shape, device = w.device) 
            normal = normal * sigma + mean
            output = torch.mul(w, normal)
            return output
        
        def uniform_fluction(module, input, output):        # uniformly distributed disturbance
            left, range = self.arg1, self.arg2
            w = output
            uniform = torch.rand(w.shape, device = w.device)
            uniform = uniform*range + left
            output = torch.mul(w,uniform)
            return output
        
        # def negative_fluction(self, module, input, output):
        #     mean, sigma  = self.arg1, self.arg2
        #     noise_client = torch.randn_like(output) * sigma + mean
        #     # Subtract larger Gaussian noise (sigma * 2 for example)
        #     noise_neg = torch.randn_like(output) * (sigma_global * sigma_neg) + mean
        #     output = output * noise_client - output * noise_neg
        #     return output
        
        # def negative_fluction(self, module, input, output, sigma_global, sigma_neg):
        #     mean, sigma  = self.arg1, self.arg2
        #     noise_client = torch.randn_like(output) * sigma + mean
        #     # Subtract larger Gaussian noise (sigma * 2 for example)
        #     noise_neg = torch.randn_like(output) * (sigma_global * sigma_neg) + mean
        #     output = output * noise_client - output * noise_neg
        #     return output

        self.fault_types = { #choose types
            'uniform':uniform_fluction,
            'Gaussian':normal_fluction,
            'Negative Feedback': negative_fluction
            }
        self.get_hook(model, self.fault_types[self.fault_type])
        
class InjectWeight():
    def __init__(self, model, fault_type='normal_flu', arg1=0, arg2=0, layer_mask=None) -> None:
        self.model = model
        self.fault_type = fault_type
        self.layer_mask = layer_mask
        self.arg1 = arg1 
        self.arg2 = arg2
        self.fault_types = { #choose types
        'uniform_flu'        :self.uniform_fluction,
        'normal_flu'         :self.normal_fluction,
        } 
    def get_layers(self, ):
        real_layers=[]
        layers = list(self.model.modules())
        for layer in layers:
            if len(list(layer.children()))==0 and len(list(layer.parameters())) :
                    real_layers.append(layer)
            if len(list(layer.children())) and len(list(list(layer.children())[0].parameters()))==0  and len(list(layer.parameters())):
                    real_layers.append(layer)
        return real_layers
    
    def expand_mask(self, model):   
        layers = self.get_layers()
        if(self.layer_mask is None):
            self.layer_mask = 1+np.zeros(len(layers), dtype='bool')
        new_mask = self.layer_mask
        return new_mask

    def transform_layers(self, model, function):              
        mask = self.expand_mask(model)
        layers = self.get_layers()
        for l,m in zip(layers, mask):
            if m:
                v = function(l)    # Must mutate the layer in-place       
                assert v is None, 'Layer transform function must do their work in-place.'
        return model
    
    def update_layer(self, layer, new_data):           #checked  
        layer.weight.data = new_data.cuda()
    
    def transform_weights(self, model, function):  
        def sub_transform(layer):
            new_weights = function(layer.weight.data)
            self.update_layer(layer, new_weights)
            return None # Mutation is in-place
        return self.transform_layers(model, sub_transform)
    
    def normal_fluction(self, layer):          ## Gaussian noise
        mean = self.arg1
        sigma  = self.arg2
        w = layer.weight.data
        normal = torch.randn(w.shape, device = w.device)
        normal = normal * sigma + mean
        layer.weight.data = torch.mul(w,normal)

    def uniform_fluction(self, layer):        # uniformly distributed disturbance
        left = self.arg1
        range = self.arg2
        w = layer[1].data
        uniform = torch.rand(w.shape, device = w.device)
        uniform = uniform*range + left
        layer[1].data = torch.mul(w,uniform)
                  
    def __call__(self,):        
        self.transform_layers(self.model, self.fault_types[self.fault_type])


class InjectWeightNoise:
    def __init__(self, model, noise_level=0.1):
        self.model = model
        self.noise_level = noise_level
        self.max_weight_values = [param.data.abs().max().item() for param in model.parameters()]
        self.current_weights = [param.data.clone() for param in model.parameters()]

    def add_noise_to_weights(self):
        for i, weight in enumerate(self.current_weights):
            noise = torch.randn_like(weight) * self.noise_level * self.max_weight_values[i]
            self.current_weights[i] += noise

    def update_model_weights(self):
        for param, new_weight in zip(self.model.parameters(), self.current_weights):
            param.data = new_weight
            
class InjectDroupout:
    def __init__(self, model, dropout_prob=0.1):
        self.model = model
        self.dropout_prob = dropout_prob
        self.inject_dropout()

    def inject_dropout(self):
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                # add Dropout to linear
                setattr(self.model, name, nn.Sequential(module, nn.Dropout(self.dropout_prob)))
            elif isinstance(module, nn.Conv2d):
                # add Dropout to conv
                setattr(self.model, name, nn.Sequential(module, nn.Dropout2d(self.dropout_prob)))

   
