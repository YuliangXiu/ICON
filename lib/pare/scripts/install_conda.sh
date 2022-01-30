#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=pare-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7.3

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

sudo apt-get install libturbojpeg
pip install -r requirements.txt