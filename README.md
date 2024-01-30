# DeepONets in JAX

Code for DeepONets (https://arxiv.org/abs/1910.03193) in JaX 

## Roadmap

### Models
#### Planned
* vanilla stacked DeepONets with Bias
* Extension to multiple outputs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)
* physics-informed DeepONets (https://arxiv.org/abs/2103.10974 ,  https://arxiv.org/abs/2207.05748)

#### Potential Extensions
* separable PI DeepONets (see separable PINNs: https://jwcho5576.github.io/spinn.github.io/, https://arxiv.org/abs/2306.15969)
* Hard Enforcement of Dirichlet BCs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)

### Data/Equations
Project is focused on later use of (physics-informed) DeepONets for solving problems in porous media. Currently available are:
* Biot's equation of consolidation in 1D

Upcoming examples:
* theory of porous media (TPM) for consolidation in 1D
* 2D Consolidation using Biot / TPM

#### Note on dimensions
Please note that all problems are spatiotemporal problems and we denote spatial dimensions here. Hence, Biot's equation for consolidation in 1D take both vertical position z and time t as input.

## References

Reference Codes: 
* https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
* https://github.com/lululxvi/deeponet
* https://github.com/lu-group/deeponet-fno

## Open Questions
* stacked vs. unstacked DeepONets?
* Bias or no Bias?
* Which strategy for multiple ($n$) outputs?
  * n independent DeepONets
  * split branch and trunk in n groups
  * split branch in n groups, share trunk
  * share branch, split trunk in n groups