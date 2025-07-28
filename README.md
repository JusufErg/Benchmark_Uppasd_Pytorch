# Benchmarking Pytorch Optimizer vs UppASD the Atomistic Spin Hamiltonian. 



## Introduction



## Theory



## File description

parser.py —— This parser parses the exchange file (jij), DMI-file (dmdata), restart.SCsuf_T.out which contains all the optomized spins from UppASD. Only the parsed restart... file is saved in data for later use as parsed_restart.csv. The rest are used as needed, ie. the function is called upon.

hamiltonian.py —— This file contains the function describing the Atomistic Spin Hamiltonian. Included are all the four terms and their coefficients, ie. Heisenber Exchange, Dzyaloshinskii–Moriya interaction (DMI), Anisotropy and External magnetic Field term.

optimizer.py —— The file optmizes the Hamiltonian with following optmizers: Adam; SGD; L-BFGS, AdamW, Adagrad, RMSprop. 

compare_spins.py —— A helper function that compares optimized spins and spins from parsed_restart.csv file. 

benchmark_runner.py —— Runs a single seed of the needed 

batch_benchmark.py ——

/data —— a file to store all data and parsed restart file. 

/SkyrmionLattice —— copied file from UppASD code that runs simulation. The files used to run this particular code are given. 

## Use


## Results



## Acknowledgements



## Licence 

