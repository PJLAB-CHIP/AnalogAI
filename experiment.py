import subprocess
import os


if __name__ == '__main__':
    exp_settings_files = os.listdir('./exp')
    exp_settings_files.remove('deprecated')
    for file_name in exp_settings_files:
        print(file_name)
        model_name = file_name.split('_')[0]
        data_name = file_name.split('_')[1]
        print(model_name,data_name)
        if model_name == 'vgg11':
            device = 'cuda:3'
        elif model_name == 'vgg16':
            device = 'cuda:4'
        elif model_name == 'resnet18':
            device = 'cuda:5'
        elif model_name == 'vit':
            device = 'cuda:6'
        elif model_name == 'mobileNet':
            device = 'cuda:7'
        command = [
            'python',
            'train_fedprox.py',
            '--config',
            file_name,
            '--device',
            device
        ]
        if data_name == 'cifar10':
            subprocess.run(command)