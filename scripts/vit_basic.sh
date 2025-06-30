export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=2 python train_basic.py --config vit_basic.yml 2>&1 |tee ./log/vit_basic.log

