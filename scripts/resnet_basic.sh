export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=1 python train_basic.py --config resnet_basic.yml 2>&1 | tee ./log/resnet_basic.log

