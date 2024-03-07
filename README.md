# RKcompress

Contains the MATLAB code for the memory-efficient computation of f(A)b for a Hermitian matrix A, using the algorithm presented in [1].

The main function is `RKcompress_fAb`, with additional helper functions contained in the `utils` folder. 

This repository additonally contains the MATLAB code to reproduce the numerical experiments in [1]:
- `demo_expm.m`: Figure 1 and Table 1
- `demo_markov1.m`: Figure 2, Table 2 and Table 3
- `demo_markov2.m`: Table 4
- `demo_lossorth.m`: Figure 3

Some functions required to run the other low-memory methods that we compare against `RKcompress_fAb` are contained in the `utils-comparison` folder; the output of the experiments is saved in the `output-data` folder.

[1] Angelo A. Casulli, Igor Simunec, A low-memory Lanczos method with rational Krylov compression for matrix functions, to appear on arXiv (2024).

## Dependencies

There are no external dependencies for the main function `RKcompress_fAb`. 
The demo scripts have the following dependencies, required for the comparison against some other low-memory methods:
- `chebfun` https://www.chebfun.org/ (for the AAA algorithm for rational approximation, used in the multishift CG algorithm)
- `funm_quad` http://www.guettel.com/funm_quad/ (for comparison against the `funm_quad` algorithm)

