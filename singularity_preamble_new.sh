#!/bin/bash
# ENERGYPLUS_VERSION=9.5.0
# ENERGYPLUS_INSTALL_VERSION=9-5-0
# ENERGYPLUS_SHA=de239b2e5f

# # Argument for Sinergym extras libraries
# SINERGYM_EXTRAS=[extras]

# # Argument for choosing Python version
# PYTHON_VERSION=3.10

# ENERGYPLUS_VERSION=$ENERGYPLUS_VERSION
# ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
# ENERGYPLUS_SHA=$ENERGYPLUS_SHA

# # This should be x.y.z, but EnergyPlus convention is x-y-z
# ENERGYPLUS_INSTALL_VERSION=$ENERGYPLUS_INSTALL_VERSION
# EPLUS_PATH=./EnergyPlus-$ENERGYPLUS_INSTALL_VERSION

# # Downloading from Github
# # e.g. https://github.com/NREL/EnergyPlus/releases/download/v9.5.0/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu18.04-x86_64.sh
# ENERGYPLUS_DOWNLOAD_BASE_URL=https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
# ENERGYPLUS_DOWNLOAD_FILENAME=EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu18.04-x86_64.sh
# ENERGYPLUS_DOWNLOAD_URL=$ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME

# # Collapse the update of packages, download and installation into one command
# # to make the container smaller & remove a bunch of the auxiliary apps/files
# # that are not needed in the container
# BCVTB_PATH=./bcvtb


# export BCVTB_PATH EPLUS_PATH ENERGYPLUS_VERSION ENERGYPLUS_INSTALL_VERSION

# apt-get update && apt-get upgrade -y \
# && apt-get install -y ca-certificates curl libx11-6 libexpat1 \
# #Energyplus installation
# && curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
# && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
# && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
# && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
# && cd ./EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
# && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater \
# # # Remove the broken symlinks
# # && cd ./bin find -L . -type l -delete \
# # BCVTB installation
# && echo "Y\r" | apt-get install default-jre openjdk-8-jdk \ 
# && apt-get install -y git wget iputils-ping \
# && wget http://github.com/lbl-srg/bcvtb/releases/download/v1.6.0/bcvtb-install-linux64-v1.6.0.jar \
# && yes "1" | java -jar bcvtb-install-linux64-v1.6.0.jar \
# && cp -R 1/ $BCVTB_PATH && rm -R 1/

# echo "Post install stuffs!"


# rm python 
# ln -s /.env/python python
# chmod +x python
# alias python=./python
# python -m pip install -e sinergym[extras]
# # python -m pip install gym==0.24.1
# export LD_LIBRARY_PATH=/global/scratch/users/djang/ActiveRL/libs:/global/software/sl-7.x86_64/modules/langs/gcc/12.1.0/lib64:${LD_LIBRARY_PATH}
export PATH=/home/miniconda/envs/ActiveRL/bin/:/global/home/users/$USER/.conda/envs/ActiveRL/bin:/global/home/users/$USER/.local/bin:$PATH
# export PYTHONHOME=/global/home/users/djang/.conda/envs/ActiveRL/bin

# export BCVTB_PATH=/global/scratch/users/$USER/ActiveRL/bcvtb
# export EPLUS_PATH=/global/scratch/users/$USER/ActiveRL/EnergyPlus-9-5-0
# alias python=/global/home/users/djang/.conda/envs/ActiveRL/bin/python
PYTHON_PATH=/home/miniconda/envs/ActiveRL/bin/python
# source /home/miniconda/bin/activate \
# && conda activate ActiveRL \
# && which python \
alias python=$PYTHON_PATH
$PYTHON_PATH -m pip install -e gym-simplegrid \
&& $PYTHON_PATH -m pip install dm_control==1.0.9 \
&& $PYTHON_PATH -m pip install dm2gym==0.2.0 \
&& $PYTHON_PATH -m pip install gym==0.24.1
