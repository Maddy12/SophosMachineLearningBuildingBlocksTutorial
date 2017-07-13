# SophosMachineLearningBuildingBlocksTutorial
A tutorial on how to build an artificial neural network model based on URL data.

## Getting Started
The first thing you will need is python. If you have no experience with python, a good start is to download Anaconda which provides you with some starter packages and a handy IDE: https://www.continuum.io/downloads. The code illustrated in this example is from Python 2.7, so it would be advisable to download the 2.7 Python version of Anaconda. The IDE that comes with Anaconda is called Spyder.

### Prerequisites
There are several packages you will need to install prior to using the code. Tensorflow and Python 2.7 in addition to the following Python packages include: numpy, pandas, baker, sklearn, mmh3, nltk, matplotlib, and keras. 

### Installing
Before installing the machine learning package Keras, you must install its dependencies. Instructions on how to install Tensorflow can be found here: https://www.tensorflow.org/install/. Instructions on how to do so for Anaconda is under the Windows installation page: https://www.tensorflow.org/install/. 

Once you have tensorflow installed and set up, there are a few more dependencies that need to be installed. From the command line, you use the command 'pip install'. This also applies to the Spyder python
```
pip install numpy pandas baker sklearn mmh3 nltk matplotlib keras
```
From the Anaconda command line interface, you use 'conda install'
```
conda install numpy pandas baker sklearn mmh3 nltk matplotlib keras
```
When using Spyder, either method works from a command prompt. 

### Running the Model
To run the code, use 'python' then our function 'compare', the parameters you are changing prefixed with '--' and the value:
```
python compare --filepath "<path where the data is stored>" --n 
```
The results of the model will be stored in the filepath you pass. 
