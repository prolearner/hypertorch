# HyperTorch

Lightweight flexible research-oriented package to compute  hypergradients in [PyTorch](https://github.com/pytorch/pytorch).

## What is an hypergradient?
Given the following bi-level problem.

![bilevel](./resources/bilevel.svg)

We call **hypergradient** the following quantity.

![hypergradient](./resources/hypergradient.svg)

Where:
* ![outerobjective](./resources/outer_objective.svg)
is called the `outer objective` (e.g. the validation loss).
* ![Phi](./resources/Phi.svg) is called the `fixed point map` (e.g. a gradient descent step or the state update function in a recurrent model)
* finding the solution of the fixed point equation ![fixed_point_eq](./resources/fixed_point_eq.svg) is referred to as the `inner problem`. This can be solved by repeatedly applying the fixed point map or using a different inner algorithm.


## Quickstart

#### hyperparameter optimization
See this [notebook](https://github.com/prolearner/hypertorch/blob/master/examples/logistic_regression.ipynb), where we show how to compute the hypergradient to optimize the regularization parameters of a simple logistic regression model.

#### meta-learning
[`examples/iMAML.py`](https://github.com/prolearner/hypertorch/blob/master/examples/iMAML.py) shows an implementation of the method described in the paper [Meta-learning with implicit gradients](https://arxiv.org/abs/1909.04630). The code uses [higher](https://github.com/facebookresearch/higher) to get stateless version of torch nn.Module-s and [torchmeta](https://github.com/tristandeleu/pytorch-meta) for meta-dataset loading and minibatching.

#### [equilibrium-models](https://arxiv.org/abs/1909.01377) 
This [notebbook](https://github.com/prolearner/hypertorch/blob/eqm_example/examples/Equilibrium%20models%20(RNN-style%20model%20on%20MNIST).ipynb) shows how to train a simple equilibrium network with "RNN-style" dynamics.


MORE EXAMPLES COMING SOON

## Use cases

Hypergadients are useful to perform
- gradient-based hyperparamter optimization
- meta-learning
- training models that use an internal state (some types of RNNs and GNNs, Deep Equilibrium Networks, ...) 

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
the inner problem. This allows to optimize the inner solver parameters such as the learning rate and momentum.

Methods in this class are:
- `reverse_unroll`: computes the approximate hypergradient by unrolling the entire computational graph of the update dynamics for solving the inner problem. The method is essentially a wrapper for standard backpropagation. IMPORTANT NOTE: the weights must be non-leaf tensors obtained through the application of "PyThorch differentiable" update dynamics (do not use built-in optimizers!). NOTE N2.: this method is memory hungry!
- `reverse`: computes the hypergradient as above but uses less memory. It uses the trajectory information and recomputes all other necessary intermediate variables in the backward pass. It requires the list of past weights and the list of `callable` update mappings applied during the inner optimization.

### Approximate Implicit Differentiation methods:
These methods approximate the hypergradient equation directly by:
 * Using an approximate solution to the inner problem instead of the true one.
 * Computing an approximate solution to the linear system `(I-J)x_star = b`, where `J` and  `b` are respectively the transpose of the jacobian of the fixed point map and the gradient of the outer objective both w.r.t the inner variable and computed on the approximate solution to the inner problem.
 
 Since computing and storing `J` is usually unfeasible, these methods exploit `torch.autograd` to compute the Jacobian-vector product `Jx` efficiently. Additionally, they do not require storing the trajectory of the inner solver, thus providing a potentially large memory advantage over iterative differentiation. These methods are not suited to optimize the parameters of the inner solver like the learning rate.

Methods in this class are:
- `fixed_point`: it approximately solves the linear system by repeatedly applying the map `T(x) = Jx + b`. NOTE: this method converges only when the fixed point map and consequently the map `T` are contractions.        
- `CG`: it approximately solves the linear system with the conjugate gradient method. IMPORTANT N0TE: `I-J` must be symmetric and positive definite  for this to work!
- `CG_normal_eq`: As above, but uses conjugate gradient on the normal equations (i.e. solves `J^TJx = J^Tb` instead) which works also when`I-J` is not symmetric and positive definite. NOTE: the  cost per iteration can be much higher than the other methods.
- `stoch_AID`: General method that can use any `torch.optim.Optimizer` to solve the linear system. See Algorithm 1 in [Grazzi et al. 2022](https://arxiv.org/abs/2202.03397) for more details. 
## Cite

If you use this code, please cite [our paper](https://arxiv.org/abs/2006.16218)
```
@inproceedings{grazzi2020iteration,
  title={On the Iteration Complexity of Hypergradient Computation},
  author={Grazzi, Riccardo and Franceschi, Luca and Pontil, Massimiliano and Salzo, Saverio},
  journal={Thirty-seventh International Conference on Machine Learning (ICML)},
  year={2020}
}
```
