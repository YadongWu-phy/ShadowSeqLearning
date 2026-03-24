# ShadowSeqLearning
Numerical results for the paper "Designing Shadow Tomography Protocols by Reinforcement Learning"

The code is written in the `Julia` programming language and uses the `ITensors.jl` package for quantum circuit evolution, and the `Flux.jl` package for machine learning optimization.

## Calculation Scripts
[EntFeaOpt.jl](EntFeaOpt.jl) and [EFShadow.jl](EFShadow.jl) are used to optimize the entanglement feature of random two-qubit gates using the gradient descent method.

[RLGateSeq.jl](RLGateSeq.jl) is used to train the gate-sequence generator with reinforcement learning. The generator is built with a vanilla RNN, and its parameters are optimized using the policy gradient method. The "environment" of the RL is [Shadow.jl](Shadow.jl), and the "agent" of the RL is [RNNCause.jl](RNNCause.jl).

## Data
Data required to reproduce the figures in "Designing Shadow Tomography Protocols by Reinforcement Learning":

`N3k3L2.CSV`, `N4k4L3.CSV`, `N5k5L4.CSV` illustrate the entanglement feature optimization process for different system sizes and layers.

`RLdataEven.zip`, `RLdataOdd.zip` display the numerical results of the reinforcement learning process with a training set consisting of Pauli operators of 'Even' and 'Odd' size.
