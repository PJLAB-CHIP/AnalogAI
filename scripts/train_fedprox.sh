python train_fedprox.py \
    --config resnet18_cifar10_client_5_epoch_1_T_F_noise_0.1_0.5_0.1_0.1_F.yml \
    --device cuda:1 \
    --comm_round 50 \
    # --use_momentum True \
    # --use_foundation_model True \