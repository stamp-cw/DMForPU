export CUDA_VISIBLE_DEVICES=1

python3 main.py --config sqd_lstm_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config punet_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config sqd_lstm_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config restormer_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config uformer_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 200

python3 main.py --config punet_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config dlpu_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config punet_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

python3 main.py --config u3net_synpu_128_mid.yaml --mode sample_model --sampling_from_epoch 100

#insardlpu
python3 main.py --config dlpu_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config punet_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config restormer_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config sqd_lstm_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config u3net_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config uformer_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100

#synpu
python3 main.py --config dlpu_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config punet_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config restormer_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config sqd_lstm_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config u3net_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config uformer_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100

#synpu noise
python3 main.py --config dlpu_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config punet_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config restormer_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config sqd_lstm_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config u3net_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100
python3 main.py --config uformer_synpu_128_big.yaml --mode sample_model --sampling_from_epoch 100