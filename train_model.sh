export CUDA_VISIBLE_DEVICES=0

python3 main.py --config punet_synpu_128_mid.yaml --mode train_model --training_from_scratch

python3 main.py --config sqd_lstm_synpu_128_mid.yaml --mode train_model --training_from_scratch


python3 main.py --config restormer_synpu_128_mid.yaml --mode train_model --training_from_scratch

python3 main.py --config uformer_synpu_128_mid.yaml --mode train_model --training_from_scratch