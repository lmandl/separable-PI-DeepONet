# DeepONets in JAX

Code for Physics-Informed DeepONets (https://arxiv.org/abs/1910.03193) in JAX 

Authors: Somdatta Goswami, Luis Mandl, Lena Lambers, Tim Ricken

### Models

#### Implemented
* physics-informed DeepONets (https://arxiv.org/abs/2103.10974 ,  https://arxiv.org/abs/2207.05748)
* separable PI DeepONets (see separable PINNs: https://jwcho5576.github.io/spinn.github.io/, https://arxiv.org/abs/2306.15969)
* Burgers Equation (Data taken from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets)
* Biot's Equation for Consolidation
* Heat equation
* Poisson's equation with random field source and CNN as branch network

## Dependencies

We have tested and validated this project with the following versions:

- **Python**: 3.11
- **CUDA**: 12.4
- **cuDNN**: 8.9
- **tqdm**: 4.66.2
- **torch**: 2.2.2
- **scipy**: 1.12.0
- **orbax-checkpoint**: 0.5.11
- **optax**: 0.2.2
- **matplotlib**: 3.8.3
- **jax[cuda12]**: 0.4.25
- **jaxlib**: 0.4.25+cuda12.cudnn89
- **flax**: 0.8.3

These versions were specifically chosen for compatibility. Using different versions might result in compatibility issues. To replicate our environment, ensure you use these exact versions or later.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## References

Reference Codes: 
* https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
* https://github.com/lululxvi/deeponet
* https://github.com/lululxvi/deepxde
