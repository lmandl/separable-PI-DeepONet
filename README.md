## **Separable Physics-Informed DeepONet: Breaking the Curse of Dimensionality in Physics-Informed Machine Learning**
[Luis Mandl](https://scholar.google.com/citations?user=sieD_UMAAAAJ&hl=en&oi=ao), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en), [Lena Lambers](https://scholar.google.com/citations?hl=en&user=QoL0pwoAAAAJ) and [Tim Ricken](https://scholar.google.com/citations?hl=en&user=TcWVyg8AAAAJ)

## Novelty
In this work, we introduce the Separable physics-informed DeepONet (Sep-PI-DeepONet). This framework employs a factorization technique, utilizing sub-networks for individual one-dimensional coordinates, thereby reducing the number of forward passes and the size of the Jacobian matrix required for gradient computations. By incorporating forward-mode automatic differentiation (AD), we further optimize computational efficiency, achieving linear scaling of computational cost with discretization density and dimensionality, making our approach highly suitable for high-dimensional PDEs. We demonstrate the effectiveness of Sep-PI-DeepONet through three benchmark PDE models: the viscous Burgers’ equation, Biot’s consolidation theory, and a parametrized heat equation. Our framework maintains accuracy comparable to the conventional PI-DeepONet while reducing training time by two orders of magnitude. Notably, for the heat equation solved as a 4D problem, the conventional PI-DeepONet was computationally infeasible (estimated 289.35 hours), while the Sep-PI-DeepONet completed training in just 2.5 hours. These results underscore the potential of Sep-PI-DeepONet in efficiently solving complex, high-dimensional PDEs, marking a significant advancement in physics-informed machine learning.

## Framework

<p align="center">
  <img src="https://github.com/lmandl/separable-PI-DeepONet/blob/master/Schematic.pdf" alt="Sep-PI-DeepONet" width="600"/>
  <br/>
  <strong>
</p>

#### Examples Implemented
* Burgers Equation (Data taken from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets)
* Biot's Equation for Consolidation
* Heat equation
* Poisson's equation with random field source and CNN as branch network

For physics-informed DeepONet, we have referred to the following works: https://arxiv.org/abs/1910.03193, https://arxiv.org/abs/2103.10974, and https://arxiv.org/abs/2207.05748
The idea of separable PI DeepONets is motivated by separable PINNs: https://jwcho5576.github.io/spinn.github.io/ and https://arxiv.org/abs/2306.15969

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```
These versions were specifically chosen for compatibility. Using different versions might result in compatibility issues. To replicate our environment, ensure you use these exact versions or later.

