# Store current venv
initVenv=$VIRTUAL_ENV


python3.6 -m pip install pandas
python3.6 -m pip install matplotlib
python3.6 -m pip install scipy
# python3.6 -m pip install dtaidistance  # DTW metrics
python3.6 -m pip install antropy       # Spectral entropy metrics
python3.6 -m pip install statsmodels   # For AR


# Ev. install venv
python3.6 -m pip install virtualenv

##########################################
# TimeGAN
##########################################


# Move in TimeGAN folder
cd Algorithms
cd TimeGAN

# Create venv
python3.6 -m virtualenv venv

# Activate virtual environment
. venv/bin/activate

# Install packages
python3.6 -m pip install tensorflow==1.15.0
python3.6 -m pip install numpy
python3.6 -m pip install sklearn
python3.6 -m pip install matplotlib


# Deactivate virtual environment
deactivate

# Return to main dir
cd ..
cd ..



##########################################
# BasicGAN
##########################################

# Move in BasicGAN folder
#cd BasicGAN

# Create venv
#python3.6 -m virtualenv venv

# Activate virtual environment
#. venv/bin/activate

# Install packages
#python3.6 -m pip install tensorflow==2.4.1
#python3.6 -m pip install torch
#python3.6 -m pip install pandas
#python3.6 -m pip install sklearn
#python3.6 -m pip install matplotlib

# Deactivate virtual environment
#deactivate

# Return to main dir
#cd ..


##########################################
# tsgen
##########################################

# Move in TimeGAN folder
cd Algorithms
cd tsgen

# Create venv
python3.6 -m virtualenv venv

# Activate virtual environment
. venv/bin/activate


python3.6 -m pip install numpy
python3.6 -m pip install pandas
python3.6 -m pip install matplotlib
python3.6 -m pip install https://files.pythonhosted.org/packages/86/9f/be0165c6eefd841e6928e54d3d083fa174f92d640fdc52f73a33dc9c54d1/tensorflow-1.4.0-cp36-cp36m-manylinux1_x86_64.whl
python3.6 -m pip install sugartensor==1.0.0.2
python3.6 -m pip install sklearn
python3.6 -m pip install future
python3.6 -m pip install lshashpy3




# Deactivate virtual environment
deactivate

# Return to main dir
cd ..
cd ..

# DBA
sudo apt install default-jdk

# Compile java
cd Algorithms
cd DBA
javac -cp commons-math3-3.6.1/commons-math3-3.6.1.jar TimeSeriesGenerator.java Tools.java Dba.java 
cd ..
cd ..

# Anomalies Injection
python3.6 -m pip install -U Pillow==8.2.0

# Filter
python3.6 -m pip install pykalman


# BasicGAN3072
python3.6 -m pip install torchvision
python3.6 -m pip install wfdb
python3.6 -m pip install tqdm
python3.6 -m pip install opencv-python

# Install tsgen packages
#sudo sh tsgen/install_all_linux.sh
