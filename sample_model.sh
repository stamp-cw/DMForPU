export CUDA_VISIBLE_DEVICES=1
python3 main.py --config config_unet_test.yaml --mode sample_model --sampling_from_epoch 100
