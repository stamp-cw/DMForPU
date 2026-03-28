export CUDA_VISIBLE_DEVICES=0

python3 main.py --config wav_synpu_128_mid.yaml --mode train --training_from_scratch --debug
python3 main.py --config wav_synpu_cut32_mid.yaml --mode train --training_from_scratch --debug
python3 main.py --config wav_synpu_cut32_test.yaml --mode train --training_from_scratch --debug


CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --config wav_synpu_cut32_mid.yaml --mode train --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --config wav_synpu_128_mid.yaml --mode train_multi --training_from_scratch



CUDA_VISIBLE_DEVICES=1,3 accelerate launch main.py --config wav_synpu_cut32_mid.yaml --mode train_multi --training_from_scratch


CUDA_VISIBLE_DEVICES=1,3 accelerate launch main.py --config wav_synpu_128_mid.yaml --mode train_multi --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config mch_synpu_128_mid.yaml --mode train_multi --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config wav_synpu_128_mid.yaml --mode train_multi --training_from_scratch

CUDA_VISIBLE_DEVICES=1,3 accelerate launch main.py --config fdu_synpu_128_mid.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_dlpu_256_big.yaml --mode train_multi --training_from_scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_noise_synpu_128_big.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_synpu_128_big.yaml --mode train_multi --training_from_scratch

# abla
#synpu
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_synpu_128_big_v1.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_synpu_128_big_v2.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_synpu_128_big_v3.yaml --mode train_multi --training_from_scratch
#insardlpu
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_dlpu_256_big_v1.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_dlpu_256_big_v2.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_dlpu_256_big_v3.yaml --mode train_multi --training_from_scratch

# pred mode
#synpu
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_synpu_128_big_v_pred.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_synpu_128_big_e_pred.yaml --mode train_multi --training_from_scratch
#insardlpu
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_dlpu_256_big_v_pred.yaml --mode train_multi --training_from_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --config fdu_dlpu_256_big_e_pred.yaml --mode train_multi --training_from_scratch
