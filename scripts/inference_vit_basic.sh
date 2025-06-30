export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1
python inference.py \
    --config inf_vit_basic.yml \
