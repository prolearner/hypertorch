# hypergrad

Lightweight research-oriented package to compute approximate hypergradients in PyTorch.

## What is an hypergradient?
Given the following bi-level problem.

<img src="https://latex.codecogs.com/gif.latex?\large&space;\min_{\lambda}&space;f(\lambda)=E(w(\lambda),\lambda),\quad&space;w(\lambda)=\Phi(w(\lambda),\lambda)." title="\large \min_{\lambda} f(\lambda)=E(w(\lambda),\lambda),\quad w(\lambda)=\Phi(w(\lambda),\lambda)." />

We call **hypegradient** the following quantity.

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)." title="\large \nabla f(\lambda) = \nabla_2 E(w(\lambda), \lambda) + \partial_2 \Phi(w(\lambda), \lambda)^\top (I- \partial_1 \Phi(w(\lambda), \lambda)^\top)^{-1} \nabla_1 E (w(\lambda), \lambda)." /></a>

* <img src="https://latex.codecogs.com/gif.latex?\large&space;E(w,\lambda)" title="\large E(w,\lambda)" /> is called the `outer objective` (e.g. the validation loss).
* <img src="https://latex.codecogs.com/gif.latex?\Phi(w,\lambda)" title="\Phi(w,\lambda)" /> is called the `fixed point map` (e.g. a gradient descent step or the state update function in a recurrent model)
* finding the solution of the fixed point equation <img src="https://latex.codecogs.com/gif.latex?\large&space;w(\lambda)=\Phi(w(\lambda),\lambda)" title="\large w(\lambda)=\Phi(w(\lambda),\lambda)" /> is referred to as the `inner problem`. This can be done by repeatedly applying the fixed point map or using a different inner algorithm.


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

The main methods for computing hypergradients are in the module `hypergrad/hypergradients.py`. 

All methods require as input:
- a list of tensors representing the inner variables (models' weights);
- another list of tensors for the outer variables (hyperparameters/meta-learner paramters);
- a `callable` differentiable outer objective;
- a `callable` that represents the differentiable update mapping (except `reverse_unroll`). For example this can be an SGD step.  

### Iterative differentiation methods:
These methods differentiate through the update dynamics used to solve
the inner problem.

Methods in this class are:
- `reverse_unroll`: the method computes the approximate hypergradient by unrolling the entire computational graph of the update dynamics for solving the inner problem. The method is essentially a wrapper for standard backpropagation. IMPORTANT NOTE: the weights must be non-leaf tensor obtained through the application of "PyThorch differentiable" update dynamics (do not use built-in optimizers!). NOTE N2.: this method is memory hungry!
- `reverse`: computes the hypergradient as above but uses less memory. It uses the trajectory information and recomputes all other necessary intermediate variables in the backward pass. It requires the list of past weights and the list of `callable` update mappings applied during the inner optimization.

### Approximate Implicit Differentiation methods:
These methods approximate the hypergradient equation directly by:
 * Using an approximate solution to the inner problem instead of the true one.
 * Computing an approximate solution to the linear system `(I-J)x_star = b`, where `J` and  `b` are respectively the jacobian of the fixed point map and the gradient of the outer objective both w.r.t the inner variable and computed on the approximate solution to the inner problem.
 
 Since computing and storing `J` is usually infeasible, these methods exploit `torch.autograd` to compute the Jacobian-vector product `Jx` efficiently. Additionally they do not require storing the trajectory of the inner solver, thus providing a potentially large memory advantage over iterative differentiation.

Methods in this class are:
- `fixed_point`: it approximately solves the linear system by repeatedly applying the map `T(x) = Jx + b`. NOTE: this method converges only when the fixed point map and consequently the map `T` are contractions.        
- `CG`: it approximately solves the linear system with the conjugate gradient method. IMPORTANT N0TE: `J` must be symmetric for this to work!
- `CG_normal_eq`: As above, but uses conjugate gradient on the normal equations (i.e. solves `J^TJx = J^Tb` instead) to deal with the non-symmetric case. NOTE: the cost per conjugate gradient iteration can be much higher than the other methods.

## Cite

COMING SOON

