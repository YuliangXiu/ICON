#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

mkdir -p data/smpl_related/models

# username and password input
echo -e "\nYou need to register at https://icon.is.tue.mpg.de/, according to Installation Instruction."
read -p "Username (ICON):" username
read -p "Password (ICON):" password
username=$(urle $username)
password=$(urle $password)

# SMPL (Male, Female)
echo -e "\nDownloading SMPL..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip&resume=1' -O './data/smpl_related/models/SMPL_python_v.1.0.0.zip' --no-check-certificate --continue
unzip data/smpl_related/models/SMPL_python_v.1.0.0.zip -d data/smpl_related/models
mv data/smpl_related/models/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl data/smpl_related/models/smpl/SMPL_FEMALE.pkl
mv data/smpl_related/models/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl data/smpl_related/models/smpl/SMPL_MALE.pkl
cd data/smpl_related/models
rm -rf *.zip __MACOSX smpl/models smpl/smpl_webuser
cd ../../..

# SMPL (Neutral, from SMPLIFY)
echo -e "\nDownloading SMPLify..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O './data/smpl_related/models/mpips_smplify_public_v2.zip' --no-check-certificate --continue
unzip data/smpl_related/models/mpips_smplify_public_v2.zip -d data/smpl_related/models
mv data/smpl_related/models/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl
cd data/smpl_related/models
rm -rf *.zip smplify_public 
cd ../../..

# ICON
echo -e "\nDownloading ICON..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=icon_data.zip&resume=1' -O './data/icon_data.zip' --no-check-certificate --continue
cd data && unzip icon_data.zip
mv smpl_data smpl_related/
rm -f icon_data.zip
cd ..

function download_for_training () {
    
    # SMPL-X (optional)
    echo -e "\nDownloading SMPL-X..."
    wget --post-data "username=$1&password=$2" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' -O './data/smpl_related/models/models_smplx_v1_1.zip' --no-check-certificate --continue
    unzip data/smpl_related/models/models_smplx_v1_1.zip -d data/smpl_related
    rm -f data/smpl_related/models/models_smplx_v1_1.zip

    # SMIL (optional)
    echo -e "\nDownloading SMIL..."
    wget --post-data "username=$1&password=$2" 'https://download.is.tue.mpg.de/download.php?domain=agora&sfile=smpl_kid_template.npy&resume=1' -O './data/smpl_related/models/smpl/smpl_kid_template.npy' --no-check-certificate --continue
    wget --post-data "username=$1&password=$2" 'https://download.is.tue.mpg.de/download.php?domain=agora&sfile=smplx_kid_template.npy&resume=1' -O './data/smpl_related/models/smplx/smplx_kid_template.npy' --no-check-certificate --continue
}


read -p "(optional) Download models used for training (y/n)?" choice
case "$choice" in 
  y|Y ) download_for_training $username $password;;
  n|N ) echo "Great job! Try the demo for now!";;
  * ) echo "Invalid input! Please use y|Y or n|N";;
esac