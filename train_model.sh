export CUDA_VISIBLE_DEVICES=0
python3 main.py --config config_unet_test.yaml --mode train_model --training_from_scratch
