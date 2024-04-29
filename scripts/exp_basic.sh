#! /bin/bash

# python train_fedprox.py --config resnet18_cifar10_asam.yml &
# python train_fedprox.py --config resnet18_cifar10_FGSM.yml &
# python train_fedprox.py --config resnet18_cifar10_sam.yml &
# python train_fedprox.py --config resnet18_cifar10_PGD.yml &


python inference.py --config resnet18_cifar10_asam.yml &
python inference.py --config resnet18_cifar10_FGSM.yml &
python inference.py --config resnet18_cifar10_sam.yml &
python inference.py --config resnet18_cifar10_PGD.yml &
