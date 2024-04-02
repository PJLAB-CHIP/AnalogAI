# -*- coding: utf-8 -*-
"""
created on 10.26
version: 1.0
noise_inject.py
"""
import torch
import torch.nn as nn
import numpy as np

class InjectForward():
    def __init__(self, fault_type='normal_flu', arg1=0., arg2=0., layer_mask=None) -> None:
        # self.model = model
        self.fault_type = fault_type
        self.layer_mask = layer_mask
        self.arg1 = arg1 
        self.arg2 = arg2
        self.hook_list = []
        
    def default_hook( module, input, output):
        print('hook',module)
        print('input',input)
        print('output',output)
        return output
        
    def get_layers(self, model):
        real_layers=[]
        layers = list(model.modules())
        for layer in layers:
            if len(list(layer.children()))==0 and len(list(layer.parameters())) :
                    real_layers.append(layer)
            if len(list(layer.children())) and len(list(list(layer.children())[0].parameters()))==0  and len(list(layer.parameters())):
                    real_layers.append(layer)
        return real_layers
    
    def expand_mask(self, model):          
        layers = self.get_layers(model)
        if(self.layer_mask is None):
            self.layer_mask = 1+np.zeros(len(layers), dtype='bool')
        new_mask = self.layer_mask
        return new_mask
    
    def get_hook(self, model, function=default_hook):
        mask = self.expand_mask(model)
        layers = self.get_layers(model)
        for l,m in zip(layers, mask):
            if m:
                a = l.register_forward_hook(function)
                self.hook_list.append(a)
                
    def __call__(self, model):
        
        def normal_fluction(module, input, output):          ## Gaussian noise 
            mean, sigma  = self.arg1, self.arg2
            w = output
            normal = torch.randn(w.shape, device = w.device) 
            normal = normal * sigma + mean
            output = torch.mul(w,normal)
            return output
        
        def uniform_fluction(module, input, output):        # uniformly distributed disturbance
            left, range = self.arg1, self.arg2
            w = output
            uniform = torch.rand(w.shape, device = w.device)
            uniform = uniform*range + left
            output = torch.mul(w,uniform)
            return output
        
        self.fault_types = { #choose types
            'uniform_flu'        :uniform_fluction,
            'normal_flu'         :normal_fluction,
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

   
