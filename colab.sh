# bvh-distance-queries only support cuda 11.0
# yet cuda 11.1 is the default version for colab
cd /etc/alternatives/
unlink cuda
ln -s /usr/local/cuda-11.0 cuda
cd /content

# conda environment
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
conda create --name icon python=3.8 -y
conda init bash
conda activate icon
conda config --env --set always_yes true

# install required packages
conda install -c conda-forge pyembree
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# # install pytorch3d
# conda install pytorch3d -c pytorch3d
# if conda installation failed, compile from source
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . && cd ..

cd /content/ICON
pip install -r requirements.txt --user --no-warn-script-location

