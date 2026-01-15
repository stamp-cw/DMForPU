#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#conda install -c conda-forge diffusers
#conda install -c conda-forge transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers["torch"] transformers
pip install matplotlib
pip install scipy
pip install wandb
pip install ml_collections
pip install tensorboard
pip install bitsandbytes
