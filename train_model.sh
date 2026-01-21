export CUDA_VISIBLE_DEVICES=0
python3 main.py --config config_pu_unet_test.yaml --mode train_model --training_from_scratch
python3 main.py --config config_unet_test.yaml --mode train_model --training_from_scratch

python3 main.py --config config_aux_unet_test.yaml --mode train_model --training_from_scratch
python3 main.py --config config_aux_unet_128_big.yaml --mode train_model --training_from_scratch
