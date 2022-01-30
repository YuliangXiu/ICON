#!/usr/bin/env bash

wget https://www.dropbox.com/s/aeulffqzb3zmh8x/pare-github-data.zip
unzip pare-github-data.zip
mkdir data/dataset_folders
rm pare-github-data.zip

mkdir -p $HOME/.torch/models/
mv data/yolov3.weights $HOME/.torch/models/