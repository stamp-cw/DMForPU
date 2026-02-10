#export WANDB_API_KEY=1234567890ab
#wandb login
#wandb sweep --project <project-name> <path-to-config file>
#wandb agent <sweep-ID>

#os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
#os.environ["WANDB_API_KEY"] = "local-0646baa3f9cd57d6313ce814c184e907ae775f8c"

export WANDB_API_KEY=local-0646baa3f9cd57d6313ce814c184e907ae775f8c
export WANDB_BASE_URL=http://localhost:18080

python hyper_main.py --config hyper_config.yaml