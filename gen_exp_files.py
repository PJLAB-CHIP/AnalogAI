import os
import yaml
import copy
import torch
import random

BASIC_EXP = './exp/deprecated_4_29/resnet18_cifar10_client_5_epoch_1_T_F_noise_0_0.2_0.1_0.1_F.yml'

model_set = ['resnet18','vgg11','mobileNet']
data_set = ['cifar10','mnist']
optimizer_adversarial_set = ['sam','asam','FGSM','PGD']
optimizer_set = ['sam','asam']
adversarial_set = ['FGSM','PGD']

PLATFORM_EXP = True

if __name__ == '__main__':

    with open(BASIC_EXP,'r') as file:
        basic_config = yaml.safe_load(file)
    
    if PLATFORM_EXP:
        basic_config['training']['client_num_in_total'] = 1
        basic_config['training']['use_fl'] = False
        basic_config['recovery']['noise_0']['act_inject']['use'] = False
        basic_config['recovery']['noise_0']['weight_inject']['use'] = False

    for model in model_set:
        for data in data_set:
            for op in optimizer_adversarial_set:
                config = copy.deepcopy(basic_config)
                config['data']['architecture'] = model
                config['data']['dataset'] = data
                if op == 'sam':
                    config['recovery']['optimizer']['sam'] = True
                elif op == 'asam':
                    config['recovery']['optimizer']['sam'] = True
                    config['recovery']['optimizer']['adaptive'] = True
                elif op == 'FGSM':
                    config['recovery']['adversarial']['FGSM']['use'] = True
                elif op == 'PGD':
                    config['recovery']['adversarial']['PGD']['use'] = True
                device = f"cuda:{random.randint(0,7)}"
                config['training']['device'] = str(device)
                config_file_name = f'./exp/{model}_{data}_{op}.yml'
                with open(config_file_name,'w') as file:
                    yaml.dump(config,file,default_flow_style=False)

    