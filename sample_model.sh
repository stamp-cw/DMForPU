export CUDA_VISIBLE_DEVICES=1
python3 main.py --config config_pu_unet_test.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config config_aux_unet_test.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config config_diff_aux_unet_test.yaml --mode sample_model --sampling_from_epoch 100


python3 main.py --config punet_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config sqd_lstm_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100