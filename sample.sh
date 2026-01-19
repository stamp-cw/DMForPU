export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2

python3 main.py --config config.yaml --mode sample --sampling_from_epoch 0


python3 main.py --config config_synpu_cut_32_test.yaml --mode sample --sampling_from_epoch 1000


python3 main.py --config config_synpu_128_small.yaml --mode sample --sampling_from_epoch 10