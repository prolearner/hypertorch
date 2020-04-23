# hyperTorch

Lightweight package to compute approximate hypergradients in PyTorch.

## Hypergradient
Given the following bi-level problem.

<a href="https://www.codecogs.com/eqnedit.php?latex=f(\lambda)&space;&=&space;E(w(\lambda),&space;\lambda),&space;\quad&space;w(\lambda)&space;&=&space;\Phi(w(\lambda),&space;\lambda)." target="_blank"><img src="https://latex.codecogs.com/png.latex?f(\lambda)&space;&=&space;E(w(\lambda),&space;\lambda),&space;\quad&space;w(\lambda)&space;&=&space;\Phi(w(\lambda),&space;\lambda)." title="f(\lambda) &= E(w(\lambda), \lambda), \quad w(\lambda) &= \Phi(w(\lambda), \lambda)." /></a>


We call **hypegradient** the following quantity.

<a href="https://www.codecogs.com/eqnedit.php?latex=\huge&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\huge&space;\nabla&space;f(\lambda)&space;=&space;\nabla_2&space;E(w(\lambda),&space;\lambda)&space;&plus;&space;\partial_2&space;\Phi(w(\lambda),&space;\lambda)^\top&space;(I-&space;\partial_1&space;\Phi(w(\lambda),&space;\lambda)^\top)^{-1}&space;\nabla_1&space;E&space;(w(\lambda),&space;\lambda)" title="\huge \nabla f(\lambda) = \nabla_2 E(w(\lambda), \lambda) + \partial_2 \Phi(w(\lambda), \lambda)^\top (I- \partial_1 \Phi(w(\lambda), \lambda)^\top)^{-1} \nabla_1 E (w(\lambda), \lambda)" /></a>

