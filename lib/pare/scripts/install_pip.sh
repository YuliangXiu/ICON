#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python3.7 -m venv pare-env
echo "Activating virtual environment"

source $PWD/pare-env/bin/activate

sudo apt-get install libturbojpeg
$PWD/pare-env/bin/pip install -r requirements.txt