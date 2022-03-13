urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

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
source activate icon
pip install gdown --upgrade
gdown https://drive.google.com/drive/u/1/folders/1CkF79XRaZzdRlj6eJUt4W0nbTORv2t7O -O pretrained_model --folder
cd ..
echo "PyMAF done!"

function download_pare(){
    # (optional) download PARE
    wget https://www.dropbox.com/s/aeulffqzb3zmh8x/pare-github-data.zip
    unzip pare-github-data.zip && mv data pare_data
    rm -f pare-github-data.zip
    cd ..

    echo "PARE done!"
}

function download_pixie(){

  mkdir -p data/pixie_data

  # SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
  echo -e "\nYou need to login https://icon.is.tue.mpg.de/ and register SMPL-X and PIXIE"
  read -p "Username (SMPL-X):" username
  read -p "Password (SMPL-X):" password
  username=$(urle $username)
  password=$(urle $password)
  wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './data/pixie_data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

  # PIXIE pretrained model and utilities
  wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './data/pixie_data/pixie_model.tar' --no-check-certificate --continue
  wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './data/pixie_data/utilities.zip' --no-check-certificate --continue
  cd data/pixie_data
  unzip utilities.zip
  rm utilities.zip
  cd ../../
}

read -p "(optional) Download PARE[SMPL] (y/n)?" choice
case "$choice" in 
  y|Y ) download_pare;;
  n|N ) echo "PARE Done!";;
  * ) echo "Invalid input! Please use y|Y or n|N";;
esac

read -p "(optional) Download PIXIE[SMPL-X] (y/n)?" choice
case "$choice" in 
  y|Y ) download_pixie;;
  n|N ) echo "PIXIE Done!";;
  * ) echo "Invalid input! Please use y|Y or n|N";;
esac