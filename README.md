# DeepONets in JAX

Code for DeepONets (https://arxiv.org/abs/1910.03193) in JaX 

## Roadmap

### Models
#### Planned
* vanilla DeepONets with Bias for antiderivative example (data taken from DeepXDE examples)
* extension to Biot's equation but only for 1 output (displacement u)
* Extension to multiple outputs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)
* physics-informed DeepONets (https://arxiv.org/abs/2103.10974 ,  https://arxiv.org/abs/2207.05748)

#### Potential Extensions
* separable PI DeepONets (see separable PINNs: https://jwcho5576.github.io/spinn.github.io/, https://arxiv.org/abs/2306.15969)
* Hard Enforcement of Dirichlet BCs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)

### Data/Equations
Project is focused on later use of (physics-informed) DeepONets for solving problems in porous media. Current plans are:
* Biot's equation of consolidation in 1D

Upcoming examples:
* theory of porous media (TPM) for consolidation in 1D
* 2D Consolidation using Biot / TPM

#### Note on dimensions
Please note that all problems are spatiotemporal problems and we denote spatial dimensions here. Hence, Biot's equation for consolidation in 1D take both vertical position z and time t as input.

#### Note on alignment of datasets
DeepXDE allows to differentiate between aligned and unaligned datasets. For the sake of easier use later on, the framework expects unaligned data. Hence, for the respective problems with aligned datasets, the data has to be loaded in the correct form or manipulated in the respective form. More information can be extracted from https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html#deepxde.data.triple.TripleCartesianProd. In short: inputs to both trunk and branch network as well as the respective outputs should always have the same sample/batch size.

## References

Reference Codes: 
* https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
* https://github.com/lululxvi/deeponet
* https://github.com/lu-group/deeponet-fno
* https://github.com/lululxvi/deepxde
* https://github.com/katiana22/TL-DeepONet
* https://github.com/kuangdai/deeponet-jax-bench
* https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets

## Open Questions
* stacked vs. unstacked DeepONets?
  * stacked DeepONets best according to https://www.sciencedirect.com/science/article/pii/S0045782522001207
* Bias or no Bias?
* Which strategy for multiple ($n$) outputs?
  * n independent DeepONets
  * split branch and trunk in n groups
  * split branch in n groups, share trunk
  * share branch, split trunk in n groups