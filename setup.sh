#!/bin/bash
echo "install pytho libraries"
pip install -r requirements.txt


echo "preparing packages"
apt-get update
apt-get install -y libboost-all-dev
apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev

apt-get install htop

#git clone --recursive https://github.com/Microsoft/LightGBM
#apt-get update --fix-missing

#echo "installing lightgbm..."
#cd LightGBM
##rm -r build
#mkdir build
#cd build
##cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so.1 -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
#cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..

apt-get -y install python-pip
apt-get install python3-pip

pip install setuptools numpy scipy scikit-learn -U

#make -j$(nproc)
#cd ../python-package
#python setup.py install --precompile

mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

pip install lightgbm==3.3.1 --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"

pip install pytorch-forecasting
#==0.9.0
pip install pytorch-lightning
#==1.3.8
pip install setuptools==59.5.0

'''
pip install pytorch-forecasting --upgrade
pip install pytorch-lightning --upgrade
pip uninstall pystan
pip install pystan~=2.14
#pip install fbprophet
#pip3 uninstall fbprophet
pip3 install fbprophet --no-cache-dir --no-binary :all:


''' 
 
git config --local user.email "mecatronico.lazo@gmail.com"
git config --local user.name "CristianLazoQuispe"



pip install xgboost
pip install catboost
#pip install ipywidgets
#jupyter nbextension enable --py widgetsnbextension