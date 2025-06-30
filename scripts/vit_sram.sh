export CUDA_VISIBLE_DEVICES=3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python train_basic.py --config vit_sram.yml 2>&1 | tee ./log/vit_sram.log

