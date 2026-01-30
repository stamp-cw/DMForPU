#python3 main.py --config punet_synpu_128_mid.yaml --mode train_model --training_from_scratch


CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --config punet_synpu_128_mid.yaml --mode train_multi_model --training_from_scratch