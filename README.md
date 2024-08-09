# DeepONets in JAX

Code for Physics-Informed DeepONets (https://arxiv.org/abs/1910.03193) in JAX 
Authors: Somdatta Goswami, Luis Mandl

## Roadmap

### Models

#### Implemented
* physics-informed DeepONets (https://arxiv.org/abs/2103.10974 ,  https://arxiv.org/abs/2207.05748)
* separable PI DeepONets (see separable PINNs: https://jwcho5576.github.io/spinn.github.io/, https://arxiv.org/abs/2306.15969)
* Burgers Equation (Data taken from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets)
* Biot's Equation for Consolidation
* Heat equation

#### Upcoming Tasks
* include loss weights as args
* inputs with several values per sensor in branch / factorized branch

#### Potential Extensions
* hard enforcement of dirichlet BCs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)
* modified MLP see (https://epubs.siam.org/doi/10.1137/20M1318043 and https://jwcho5576.github.io/spinn.github.io/)
* include NTK based weighting schemes and improved architecture (https://doi.org/10.1007/s10915-022-01881-0)
* L2 relative error loss functions

#### Multiple inputs to branch network
function with several values per sensor can be included which will be achieved by flattening the input in the DeepONet 

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
* https://github.com/lu-group/deeponet-fno
* https://github.com/lululxvi/deepxde
* https://github.com/katiana22/TL-DeepONet

## Open Questions
* stacked vs. unstacked DeepONets?
  * stacked DeepONets best according to https://www.sciencedirect.com/science/article/pii/S0045782522001207
    > "Four slightly different versions of DeepONet have been developed in Ref. [18], but in this study we use the stacked DeepONet with bias, which exhibits the best performance in practice among all our versions of DeepONet."
  * unstacked DeepONets best according to https://arxiv.org/abs/1910.03193
    > "Compared to stacked DeepONets, although unstacked DeepONets have larger training error, the test error is smaller, due to the smaller generalization error. Therefore, unstacked DeepONets with bias achieve the best performance. In addition, unstacked DeepONets have fewer number of parameters than stacked DeepONets, and thus can be trained faster using much less memory."
* Which strategy for multiple ($n$) outputs?