# hypergrad

Lightweight package to compute approximate hypergradients in PyTorch.

## What we mean by hypergradient~~~~~~~~
Given the following bi-level problem.

<img src="https://latex.codecogs.com/gif.latex?\large&space;f(\lambda)&space;=&space;E(w(\lambda),&space;\lambda),&space;\quad&space;w(\lambda)&space;=&space;\Phi(w(\lambda),&space;\lambda)." title="\large f(\lambda) = E(w(\lambda), \lambda), \quad w(\lambda) = \Phi(w(\lambda), \lambda)." />

We call **hypegradient** the following quantity.

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)." title="\large \nabla f(\lambda) = \nabla_2 E(w(\lambda), \lambda) + \partial_2 \Phi(w(\lambda), \lambda)^\top (I- \partial_1 \Phi(w(\lambda), \lambda)^\top)^{-1} \nabla_1 E (w(\lambda), \lambda)." /></a>

* <img src="https://latex.codecogs.com/gif.latex?E(w,&space;\lambda)" title="E(w, \lambda)" /> is called the `outer objective` (e.g. the validation loss).
* <img src="https://latex.codecogs.com/gif.latex?\Phi(w,\lambda)" title="\Phi(w,\lambda)" /> is called the `fixed point mapping`
* finding the solution of the fixed point equation <img src="https://latex.codecogs.com/gif.latex?w(\lambda)&space;=&space;\Phi(w(\lambda),&space;\lambda)." title="w(\lambda) = \Phi(w(\lambda), \lambda)." /> is referred to as the inner problem.


## Quickstart

See this [IPython book](https://github.com/prolearner/hyperTorch/blob/master/examples/logistic_regression.ipynb), where we show how to compute the hypergradient to optimize the regularization parameters of a simple logistic regression model.

MORE EXAMPLES COMING SOON

## Use cases

Hypergadients are useful to perform
- gradient-based hyperparamter optimization
- meta-learning
- training models that use an internal state (RNNs, GNNs, Deep Equilibrium Networks, ...) 
- the experiments of your next paper

## Install
Requires python 3 and PyTorch version >= 1.4.

```
git clone git@github.com:prolearner/hypergrad.git
cd hypergrad
pip install .
```
`python setup.py install` would also work.

## Implmented methods

The main methods for computing hypergradients are in the module `hyperTorch/hg/hypergradients.py`. 

All methods require as input:
- a list of tensors representing the inner variables (models' weights);
- another list of tensors for the outer variables (hyperparameters/meta-learner paramters);
- a `callable` differentiable outer objective;
- a `callable` that represents the differentiable update mapping (except `reverse_unroll`). For example this can be an SGD step.  

Currently implemented are:
- `reverse_unroll`: iterative differentiation; the method computes the approximate hypergradient by unrolling the entire computational graph of the update dynamics for solving the inner problem. The method is essentially a wrapper for standard backpropagation and needs only the outer objective. IMPORTANT NOTE: for the method to work properly, the weights be obtained through the application of a "PyThorch differentiable" optimization dynamics (do not use built-in optimizers!). NOTE N2.: this method is memory hungry!
- `reverse`: iterative differentiation; computes the hypergradient as above but it is less memory hungry. It uses the trajectory information and recomputes all other necessary intermediate variables in the backward pass. The trajectory must be passed as a list of past weights (that need to be stored in the forward pass) and `callable` update mappings.
- `fixed_point`: implicit differentiation; it computes the hypergradient by iterating the differentiated update mapping at the last inner iterate, interpreted as a fixed point equation. NOTE: good pracices to prevent divercence include checking that the update mapping is indeed a contracton.        
- `CG`: implicit differentiation; it computes the hypergradient by approximately solving a linear system arising from the hypergradient equation by conjugate gradient. As `fixed_point`, `CG` needs only infromation of the last iterate. IMPORTANT N0TE: the Jacobian of the update mapping (w.r.t. the inner variables) must be symmetric!
- `CG_normal_eq`: implicit differentiation: As above, but uses conjugate gradient on the normal equations to deal with the non-symmetric case.  

Where available, the parameter `K` controls the number of iterations of the  hypergradient approximation algorithms.
Generally speaking, higher `K` correspond to higher accuracy and higher computation time (scales linearly with `K`)

## Cite

COMING SOON

