# fetch DensePose UV data from facebookresearch/DensePose
mkdir -p data/pymaf_data/UV_data && cd data/pymaf_data/UV_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz
cd ..

# download mesh_downsampling file
wget https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz

# Model constants etc from https://github.com/nkolot/SPIN/blob/master/fetch_data.sh
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz
tar xvf data.tar.gz
mv data/* .
rm -rf data && rm -f data.tar.gz

# PyMAF pre-trained model
gdown https://drive.google.com/drive/u/1/folders/1CkF79XRaZzdRlj6eJUt4W0nbTORv2t7O -O pretrained_model --folder
cd ..
echo "PyMAF done!"

function download_pare(){
    # (optional) download PARE
    wget https://www.dropbox.com/s/aeulffqzb3zmh8x/pare-github-data.zip
    unzip pare-github-data.zip && mv data pare_data
    rm -f pare-github-data.zip

    echo "PARE done!"
}

read -p "(optional) Download PARE (y/n)?" choice
case "$choice" in 
  y|Y ) download_pare;;
  n|N ) echo "Done!";;
  * ) echo "Invalid input! Please use y|Y or n|N";;
esac