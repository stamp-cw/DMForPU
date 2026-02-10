# -*- coding: utf-8 -*-
"""
@File    : hyper_local_main.py
@Author  : 18744
@Time    : 2025/10/4 09:29
@Description : 
"""
import wandb
import yaml
from datetime import datetime
import argparse
import os

os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
os.environ["WANDB_API_KEY"] = "local-0646baa3f9cd57d6313ce814c184e907ae775f8c"



parser = argparse.ArgumentParser(description=globals()['__doc__'])
parser.add_argument('--config', type=str, required=True, help='Path to the configs file')
args = parser.parse_args()

with open(os.path.join('configs', args.config), 'r') as f:
    sweep_config = yaml.safe_load(f)

run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

sweep_id = wandb.sweep(sweep=sweep_config, project=f"sweep-{run_time}")

wandb.agent(sweep_id)
