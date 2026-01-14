#export WANDB_API_KEY=1234567890ab
#wandb login
#wandb sweep --project <project-name> <path-to-config file>
#wandb agent <sweep-ID>
python hyper_main.py --config hyper_config.yaml