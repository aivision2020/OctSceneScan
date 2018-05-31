# desc2pix
my pool side project to test privacy risk from sparse map descriptors

git submodule init
git submodule update --recursive


pip install --upgrade pip
pip install opencv-python
pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
pip install tornado==4.5 #don't know why. latest doesn't work 
pip install torchvision
pip install visdom
pip install pytest
pip install ipdb==0.10.2
pip install scikit-image


not sure which of there we need:
pip install progressbar tensorboard_logger nibabel tqdm joblib
pip install numpy scipy joblib pandas matplotlib scikit-learn scikit-image cython pyflakes pyyaml seaborn protobuf
