# DeepONets in JAX

Code for DeepONets (https://arxiv.org/abs/1910.03193) in JAX 

## Roadmap

### Models

#### Implemented
* vanilla DeepONets (stacked/unstacked) with bias and extension to multiple outputs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)
  * split trunk and/or split branch
* Burgers Equation PI (Data taken from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets)
* physics-informed DeepONets (https://arxiv.org/abs/2103.10974 ,  https://arxiv.org/abs/2207.05748)

#### Current TODOs
* Getting Burgers Equation working separable DeepONets
* Forward mode AD in separable PI DeepONets
* Consistency using x and t in Burgers Example

#### Upcoming Tasks
* Biot's equation of consolidation in 1D
* 

#### Potential Extensions
* separable PI DeepONets (see separable PINNs: https://jwcho5576.github.io/spinn.github.io/, https://arxiv.org/abs/2306.15969)
* hard enforcement of dirichlet BCs (https://www.sciencedirect.com/science/article/pii/S0045782522001207)
* modified MLP see (https://epubs.siam.org/doi/10.1137/20M1318043 and https://jwcho5576.github.io/spinn.github.io/)

#### Future Projects / Minor TODOs
* Remove purely data-driven system for easy use
* Refactor training loop into PI_DeepONet class
* Unify use of jax.vmap and jax.jit
* Visualization
* CheckPoint / Save & Load Weights
* Input Shape in Flax is inferred from last axis: Does not need to be specified beforehand,
* adjust network size accordingly

### Data/Equations
Project is focused on later use of (physics-informed) DeepONets for solving problems in porous media. Current plans are:
* Biot's equation of consolidation in 1D

Upcoming examples:
* theory of porous media (TPM) for consolidation in 1D
* 2D Consolidation using Biot / TPM

#### Note on dimensions
Please note that all problems are spatiotemporal problems and we denote spatial dimensions here. Hence, Biot's equation for consolidation in 1D take both vertical position z and time t as input.

#### Note on alignment of datasets
DeepXDE allows to differentiate between aligned and unaligned datasets. For the sake of easier use later on, the framework expects unaligned data. Hence, for the respective problems with aligned datasets, the data has to be loaded in the correct form or manipulated in the respective form. More information can be extracted from https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html#deepxde.data.triple.TripleCartesianProd. (TBD in code)

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
* batch_size or simply full_batch training?