#!/usr/bin/env bash

# conda create -n cs224n_dfp python=3.8
# conda activate cs224n_dfp

# conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
# pip install tqdm==4.58.0
# pip install requests==2.25.1
# pip install importlib-metadata==3.7.0
# pip install filelock==3.0.12
# pip install sklearn==0.0
# pip install tokenizers==0.10.1
# pip install explainaboard_client==0.0.7
# python3.8 -m venv cs224n_dfp
# source cs224n_dfp/bin/activate

# Install the required packages

# Check the version of Python
if ! command -v python3.8 &> /dev/null
then
    echo "Python 3.8 is not installed. Please install Python 3.8 and try again."
    exit
fi

python3.8 -m venv /content/cs224n_dfp
source /content/cs224n_dfp/bin/activate

pip install torch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch 
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==0.24.2
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7
pip install zipp==3.11.0
pip install idna==3.4
pip install chardet==4.0.0