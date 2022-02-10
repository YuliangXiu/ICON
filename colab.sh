# bvh-distance-queries only support cuda 11.0
# yet cuda 11.1 is the default version for colab
cd /etc/alternatives/
unlink cuda
ln -s /usr/local/cuda-11.0 cuda
cd /content

# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
conda config --env --set always_yes true
rm Miniconda3-py38_4.10.3-Linux-x86_64.sh
conda update -n base -c defaults conda -y

# conda environment setup
cd /content/ICON
conda env create -f environment.yaml
conda init bash
source ~/.bashrc
source activate icon

# install packages for colab
pip install ipykernel ipywidgets --user --no-warn-script-location

