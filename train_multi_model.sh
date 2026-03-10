#python3 main.py --config punet_synpu_128_mid.yaml --mode train_model --training_from_scratch


CUDA_VISIBLE_DEVICES=1,3 accelerate launch main.py --config punet_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch


CUDA_VISIBLE_DEVICES=1,3 accelerate launch main.py --config dlpu_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config dlpu_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config sqd_lstm_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config punet_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config u3net_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch

#insardlpu
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config punet_dlpu_256_big.yaml --mode train_multi_model --training_from_scratch