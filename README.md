# CPPND: Sparse Convolutional Network (SparseConvNet)

This program trains a sparse convolutional network (SparseConvNet) on the MNIST dataset.

The program uses the PyTorch C++ API (libtorch) for neural network development.
The MNIST dataset is loaded and used to train a [VGG-style](https://arxiv.org/abs/1409.1556)
convolutional neural network. This convolutional network has the following architecture:

vvvvvvvvvvvvvvvvvvvvv  
28 x 28 image  
.....................  
Conv3-16  
SparseMask  
Conv3-16  
Maxpool-2  
.....................  
Conv3-32  
SparseMask  
Conv3-32  
Maxpool-2  
.....................  
Conv3-64  
SparseMask  
Conv3-64  
Maxpool-2  
.....................  
Conv3-128  
SparseMask  
Conv3-128  
Maxpool-global  
.....................  
FC-128  
SparseMask  
FC-128  
SparseMask  
FC-10  
Softmax
vvvvvvvvvvvvvvvvvvvvv  

The purpose of developing the SparseConvNet is to study the effects of architecture
sparsity on the behavior of a neural network both during training and inference.
The current implementation of SparseConvNet allows for artificial sparsity via
binary masks eigen operation at each layer. This results in a specific level of
theoretical sparsity that may be helpful for experimental studies. Concrete
performance improvements with inducing sparsity requires specialized software 
and hardware implementations.

# Repository organization
This project is organized as the following:

/sparseconvnet  
|  
|--> /build  
|--> /data  
|--> /libtorch  
|--> /src

The data folder must contain the MNIST data files. This data can be downloaded 
by the download_mnist.py script as provided by the [PyTorch repository](https://github.com/pytorch/pytorch/blob/master/tools/download_mnist.py).

In the /src folder, there are two main classes that accompany the main.cpp program. 
These classes are called `DataConfig` and `SparseConvNet`.

The `DataConfig` class is a configuration class for holding all the required 
parameters for processing the data and during the training process. 
It also includes invariants with respect to such parameters.

The `SparseConvNet` class is the class for building the SparseConvNet model architecture. 
This class inherits from the PyTorch `torch::nn::Module` class and implements its 
`forward` method.

The main program, `main.cpp` includes all the necessary procedures for:
1. loading data,
2. creating batch data generators, 
3. constructing the model,
4. setting up loss functions,
5. setting up the optimizers,
6. running the forward/backward passes and updated the network weights,
7. intermittent logging,
8. calculating test metrics and printing the final results.

# Installation and Running
## Dependencies for Running Locally

* cmake >= 3.0
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* gcc/g++ >= 7
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* C++ Boost library
  * Linux: `sudo apt-get install libboost-all-dev`
  * MacOS: `brew install boost`
  * Windows: follow instructions under the [Boost Windows installation page](https://www.boost.org/doc/libs/1_55_0/more/getting_started/windows.html).
* libtorch (PyTorch C++ API) >= 1.7
  * Linux:
    * `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`
    * `unzip libtorch-shared-with-deps-latest.zip`
  * MacOS:
    * `wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip`
    * `unzip libtorch-macos-1.7.1.zip.zip`
  * Windows:
    * `wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.7.1%2Bcpu.zip`
  * Refer to [PyTorch installation docs](https://pytorch.org/cppdocs/installing.html) for more info.
* [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

## Basic Build Instructions

1. Download PyTorch libtorch and unzip by running the following under the project directory `sparseconvnet`:
   1. `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`
   2. `unzip libtorch-shared-with-deps-latest.zip`
2. Download the MNIST dataset in the `data` folder using PyTorch's utility download script by changing to the data folder `cd data` and running `python download_mnist.py` in the data data folder.
3. Make a build directory in the top level directory: `mkdir build && cd build`
4. Prepare for compile: `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..` In this case: `cmake -DCMAKE_PREFIX_PATH=/home/workspace/sparseconvnet/libtorch ..` 
5. Compile: `cmake --build . --config Release`
6. Run with command line args or default args by just running: `./main`.

## Running and Behavior
The program takes a total of 8 command line arguments as enumerated below, and
must be provided in the mentioned order. The program may be executed without
specifying any arguments, in that case the defaults in the program will be used.

When the main program is executed, the convnet training starts and the user
can expect to see the loss values as printed on the screen to decrease as the
training progresses. At the end of the program, the classification test accuracy 
is printed to the console.

Command line arguments are as below.
1. data directory (string, default "../data")
2. num epochs (integer, default 5)
3. batch size (integer, default 32)
4. image pixel value normalization mean (float, default 0.5)
5. image pixel value normalization stdev (float, default 0.5)
6. level of SparseConvNet sparsity and must be in range [0, 1) (float, default 0.2)
7. SGD learning rate (float, default 0.001)
8. num workers for loading data (integer, default 2)
