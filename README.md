# This repository contains all of the programs developed in my master's thesis, titled "Accelerating computations for dark matter direct detection experiments via neural networks and GPUs".

The "original" directory contains the first Python program given to me at the beginning of the project,
developed by my co-supervisor Einar Urdshals. In "cpp_cpu", I have developed a optimised version of
the original Python program in C++. In the "cuda" directory, the computations have been parallelised using
Nvidia's CUDA framework. The "ann" directory contains the training and validation files used to
develop an artificial neural network for predicting transition rates. The Si and Ge files contain
detector crystal data, and the "check_data" directory contains some scripts for verifying the results
of the neural network. Finally, the "data_dirs" directory contains some files for training and assessing
the neural network. Each directory contains its own README file.

