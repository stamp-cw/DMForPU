export CUDA_VISIBLE_DEVICES=1
python3 main.py --config config_vae_test.yaml --mode test_vae --sampling_from_epoch 1000
