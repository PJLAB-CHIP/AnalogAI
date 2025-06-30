export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
python train_basic.py --config mobilenet_basic.yml 2>&1 |tee ./log/mobilenet_basic.log

