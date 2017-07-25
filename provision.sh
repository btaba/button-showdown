
# https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-16-04

sudo apt-get update
sudo apt-get install python3.5

# conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# install rllab
git clone https://github.com/btaba/rllab
cd rllab
git fetch
git checkout btaba
mkdir experiments
./scripts/setup_linux.sh

# theano and lasagne version mismatch
pip install --upgrade theano
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install cached_property
conda install mkl-service
# pip install mujoco-py --no-cache
pip install -e .


cd ~
git clone https://github.com/btaba/button-showdown
cd button-showdown
pip install -r requirements.txt

pip install uwsgi


