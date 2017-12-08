

DeepJet: Repository for training and evaluation of deep neural networks for HEP
===============================================================================


Setup (CERN)
==========
It is essential to perform all these steps on lxplus7. Simple ssh to 'lxplus7' instead of 'lxplus'

Pre-Installtion: Anaconda setup (only once)
Download miniconda2
```
cd <afs work directory: you need some disk space for this!>
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
```
Please follow the installation process. If you don't know what an option does, please answer 'yes'.
After installation, you have to log out and log in again for changes to take effect.
If you don't use bash, you might have to add the conda path to your .rc file
```
export PATH="<your miniconda directory>/miniconda2/bin:$PATH"
```
This has to be only done once.


Installation:

```
mkdir <your working dir>
cd <your working dir>
git clone https://github.com/mstoye/DeepJet
cd DeepJet/environment
./setupEnv.sh deepjetLinux3.conda
```
For enabling gpu support add 'gpu' as an additional option to the last command.
This will take a while. Please log out and in again once the installation is finised.

When the installation was successful, the DeepJet tools need to be compiled.
```
cd <your working dir>
cd DeepJet/environment
source lxplus_env.sh / gpu_env.sh
cd ../modules
make -j4
```

After successfully compiling the tools, log out and in again.
The environment is set up.


Usage
==============

After logging in, please source the right environment (please cd to the directory first!):
```
cd <your working dir>/DeepJet/environment
source lxplus_env.sh / gpu_env.sh
```

Training
====

Since the training can take a while, it is advised to open a screen session, such that it does not die at logout.
```
ssh lxplus.cern.ch
<note the machine you are on, e.g. lxplus058>
screen
ssh lxplus7
```
Then source the environment, and proceed with the training. Detach the screen session with ctr+a d.
You can go back to the session by logging in to the machine the session is running on (e.g. lxplus58):

```
ssh lxplus.cern.ch
ssh lxplus058
screen -r
``` 

Please close the session when the training is finished


open Train/trainEval.py in your favorite text editor.  The top section of the file contains the relevant variables to be changed.
Input Data
Model to be used
Options to only run the training or only run the evaluation or run both
Etc.

The training is launched in the following way:
```
python trainEval.py```



