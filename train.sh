export CUDA_VISIBLE_DEVICES=0
python3 main.py --config config.yaml --mode train --training_from_scratch

python3 main.py --config config_synpu_cut_32_test.yaml --mode train --training_from_scratch

python3 main.py --config config_synpu_128_small.yaml --mode train --training_from_scratch
python3 main.py --config config_synpu_128_test_dph.yaml --mode train --training_from_scratch

python3 main.py --config config_synpu_128_test_ucn.yaml --mode train --training_from_scratch

python3 main.py --config config_synpu_128_mid_phase.yaml --mode train --training_from_scratch