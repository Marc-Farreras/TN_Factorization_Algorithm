# TN_Factorization_Algorithm

Welcome to the repository for the code used in the master's thesis project: *Long-time evolution of quantum systems with Tensor Network techniques*. This project was carried out as part of the Master's program in Quantum Science and Technologies at the University of Barcelona, in collaboration with the Barcelona Supercomputing Center (BSC), under the supervision of Dr. Stefano Carignano.

The main focus of this project was the study and characterization of a recent algorithm developed by Miguel Frías, Luca Tagliacozzo, and Mari Carmen Bañuls (arXiv:2308.04291). This algorithm enables time evolution over much longer durations than what current state-of-the-art tensor network methods can achieve.

### Repository Contents

This repository contains six Python files, each corresponding to a specific task within the algorithm:

1. **Factorization.py**: This script contains the core algorithm for decomposing the initial quantum state into fast and slow tensor components.

2. **Gradient_descent.py**: This file includes the functions necessary for implementing the gradient descent used in the heuristic truncation process.

3. **iTEBD.py**: This code performs the standard iTEBD algorithm. While it is not directly used in the main algorithm, the iMPDO_v1 and iMPDO_v2 routines inherit their core behavior from this implementation. This code has been benchmarked against the TenPy library, yielding identical results but with lower efficiency compared to TenPy. Therefore, it is recommended to use TenPy if you are primarily interested in the standard iTEBD.

4. **iTEBD_factorized.py**: This script contains the functions needed to execute the iMPDO_v2 algorithm, as described in the master's thesis. Only iMPDO_v2 is included here because it is significantly more efficient than iMPDO_v1, with similar results but much faster execution time.

5. **utils.py**: This module contains auxiliary functions that were added later to improve the overall efficiency of the code. While not essential to the algorithm, they serve as useful utilities.

6. **marc_iMPDO_v2.py**: This file serves as the main entry point for the iMPDO_v2 algorithm, orchestrating the various subroutines required to execute the full process.

This repository provides all the tools necessary to explore and reproduce the results discussed in the thesis.
