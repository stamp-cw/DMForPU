export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2

#python3 main.py --config config.yaml --mode sample --sampling_from_epoch 0
#python3 main.py --config config_synpu_cut_32_test.yaml --mode sample --sampling_from_epoch 1000
#python3 main.py --config config_synpu_128_small.yaml --mode sample --sampling_from_epoch 10

#python3 main.py --config config_synpu_128_test_dph.yaml --mode sample --sampling_from_epoch 10
#python3 main.py --config config_synpu_32_cut_mid_test_phase.yaml --mode sample --sampling_from_epoch 72 --debug

#python3 main.py --config mch_grad_synpu_cut32_test.yaml --mode sample --sampling_from_epoch 100 --debug
#python3 main.py --config mch_grad_synpu_cut32_tid.yaml --mode sample --sampling_from_epoch 100 --debug

python3 main.py --config mch_synpu_cut32_test.yaml --mode sample --sampling_from_epoch 100 --debug
python3 main.py --config mch_dlpu_cut32_big.yaml --mode sample --sampling_from_epoch 100 --debug

python3 main.py --config wav_synpu_cut32_mid.yaml --mode sample --sampling_from_epoch 100 --debug
python3 main.py --config wav_synpu_cut32_test.yaml --mode sample --sampling_from_epoch 1200 --debug

python3 main.py --config cfg_wav_synpu_cut32_test.yaml --mode sample --sampling_from_epoch 1200 --debug
python3 main.py --config cfg_wav_synpu_cut32_mid.yaml --mode sample --sampling_from_epoch 1100 --debug