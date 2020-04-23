# hyperTorch

Lightweight package to compute approximate hypergradients in PyTorch.

## Hypergradient
Given the following bi-level problem.

<img src="https://latex.codecogs.com/gif.latex?\large&space;f(\lambda)&space;=&space;E(w(\lambda),&space;\lambda),&space;\quad&space;w(\lambda)&space;=&space;\Phi(w(\lambda),&space;\lambda)." title="\large f(\lambda) = E(w(\lambda), \lambda), \quad w(\lambda) = \Phi(w(\lambda), \lambda)." />

We call **hypegradient** the following quantity.

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)." title="\large \nabla f(\lambda) = \nabla_2 E(w(\lambda), \lambda) + \partial_2 \Phi(w(\lambda), \lambda)^\top (I- \partial_1 \Phi(w(\lambda), \lambda)^\top)^{-1} \nabla_1 E (w(\lambda), \lambda)." /></a>

`hg/hypergradients` contains functions to compute an approximation to the hypergradient
given an outer objective
<img src="https://latex.codecogs.com/gif.latex?E(w,&space;\lambda)" title="E(w, \lambda)" />
a fixed point mapping 
<img src="https://latex.codecogs.com/gif.latex?\Phi(w,\lambda)" title="\Phi(w,\lambda)" />
which for most methods has to be a contraction and an approximate solution to the fixed point equation 
<img src="https://latex.codecogs.com/gif.latex?w(\lambda)&space;=&space;\Phi(w(\lambda),&space;\lambda)." title="w(\lambda) = \Phi(w(\lambda), \lambda)." />

The parameter `K` controls the number of iteration of the  hypergradient approximation algorithms.
Higher `K` correspond to higher accuracy and higher computation time (scales linearly with `K`)
