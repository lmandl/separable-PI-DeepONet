# DeepONets in JAX

Code for DeepONets (https://arxiv.org/abs/1910.03193) in JAX 

## Roadmap

### Models

#### Implemented
* Burgers Equation PI (Data taken from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets)
* physics-informed DeepONets (https://arxiv.org/abs/2103.10974 ,  https://arxiv.org/abs/2207.05748)
* separable PI DeepONets (see separable PINNs: https://jwcho5576.github.io/spinn.github.io/, https://arxiv.org/abs/2306.15969)

#### Current TODOs
* fix consistency for the use of p_ics_train, p_bcs_train, p_res_train, and p_test and make it compatible with sampling from a predefined grid

#### Upcoming Tasks
* include loss weighting as args
* save model / checkpointing
* Biot's equation of consolidation in 1D

#### Potential Extensions
* hard enforcement of dirichlet BCs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)
* modified MLP see (https://epubs.siam.org/doi/10.1137/20M1318043 and https://jwcho5576.github.io/spinn.github.io/)

### Data/Equations
Project is focused on later use of physics-informed DeepONets for solving problems. Current plans are:
* Biot's equation of consolidation in 1D
* fracture mechanics example

Upcoming examples:
* theory of porous media (TPM) for consolidation in 1D
* 2D Consolidation using Biot / TPM

#### Note on dimensions
Please note that all problems are spatiotemporal problems and we denote spatial dimensions here. Hence, Biot's equation for consolidation in 1D take both vertical position z and time t as input.

#### Multiple inputs to branch network
function with several values per sensor can be included which will be achieved by flattening the input in the DeepONet 

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
  * n independent DeepONets
  * split branch and trunk in n groups
  * split branch in n groups, share trunk
  * share branch, split trunk in n groups
* Multiple inputs to branch (e.g. two outputs (u,p) and two inputs (u, p)?)