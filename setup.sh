#!/usr/bin/env bash

conda activate cs224n_dfp || exit

conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch|| exit
pip install tqdm==4.58.0|| exit
pip install requests==2.25.1|| exit
pip install importlib-metadata==3.7.0|| exit
pip install filelock==3.0.12|| exit
pip install sklearn==0.0|| exit
pip install tokenizers==0.10.1|| exit
pip install explainaboard_client==0.0.7|| exit
pip install zipp==3.11.0|| exit
pip install idna==3.4|| exit
pip install chardet==4.0.0|| exit

## run command
cd /content/IFT6289_final_project/ || exit
# python3 classifier.py –option [pretrain/finetune] –epochs NUM_EPOCHS –lr LR –
# batch_size=BATCH_SIZE hidden_dropout_prob=RATE

python3 classifier.py --option pretrain  --lr 1e-3 --use_gpu | tee /content/assignment/pretrain.log  
python3 classifier.py --option finetune  --lr 1e-5 --use_gpu  
