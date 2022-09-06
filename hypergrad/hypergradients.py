import torch
from torch.autograd import grad as torch_grad
from torch import Tensor
from hypergrad import CG_torch
from typing import List, Callable


# noinspection PyUnusedLocal
def reverse_unroll(params: List[Tensor],
                   hparams: List[Tensor],
                   outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                   set_grad=True) -> List[Tensor]:
    """
    Computes the hypergradient by backpropagating through a previously employed inner solver procedure.

    Args:
        params: the output of a torch differentiable inner solver (it must depend on hparams in the torch graph)
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
        the list of hypergradients for each element in hparams
    """
    o_loss = outer_loss(params, hparams)
    grads = torch.autograd.grad(o_loss, hparams, retain_graph=True)
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


# noinspection PyUnusedLocal
def reverse(params_history: List[List[Tensor]],
            hparams: List[Tensor],
            update_map_history: List[Callable[[List[Tensor], List[Tensor]], List[Tensor]]],
            outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
            set_grad=True) -> List[Tensor]:
    """
    Computes the hypergradient by recomputing and backpropagating through each inner update
    using the inner iterates and the update maps previously employed by the inner solver.
    Similarly to checkpointing, this allows to save memory w.r.t. reverse_unroll by increasing computation time.
    Truncated reverse can be performed by passing only part of the trajectory information, i.e. only the
    last k inner iterates and updates.

    Args:
        params_history: the inner iterates (from first to last)
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        update_map_history: updates used to solve the inner problem (from first to last)
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
         the list of hypergradients for each element in hparams

    """
    params_history = [[w.detach().requires_grad_(True) for w in params] for params in params_history]
    o_loss = outer_loss(params_history[-1], hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params_history[-1], hparams)

    alphas = grad_outer_w
    grads = [torch.zeros_like(w) for w in hparams]
    K = len(params_history) - 1
    for k in range(-2, -(K + 2), -1):
        w_mapped = update_map_history[k + 1](params_history[k], hparams)
        bs = grad_unused_zero(w_mapped, hparams, grad_outputs=alphas, retain_graph=True)
        grads = [g + b for g, b in zip(grads, bs)]
        alphas = torch_grad(w_mapped, params_history[k], grad_outputs=alphas)

    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def fixed_point(params: List[Tensor],
                hparams: List[Tensor],
                K: int ,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True,
                stochastic=False) -> List[Tensor]:
    """
    Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).

    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        K: the maximum number of fixed point iterations
        fp_map: the fixed point map which defines the inner problem
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        tol: end the method earlier when  the normed difference between two iterates is less than tol
        set_grad: if True set t.grad to the hypergradient for every t in hparams
        stochastic: set this to True when fp_map is not a deterministic function of its inputs

    Returns:
        the list of hypergradients for each element in hparams
    """

    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)
    for k in range(K):
        vs_prev_vec = vs_vec

        if stochastic:
            w_mapped = fp_map(params, hparams)
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
        else:
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)
        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def CG(params: List[Tensor],
       hparams: List[Tensor],
       K: int ,
       fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
       outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
       tol=1e-10,
       set_grad=True,
       stochastic=False) -> List[Tensor]:
    """
     Computes the hypergradient by applying K steps of the conjugate gradient method (CG).
     It can end earlier when tol is reached.

     Args:
         params: the output of the inner solver procedure.
         hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
         K: the maximum number of conjugate gradient iterations
         fp_map: the fixed point map which defines the inner problem
         outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
         tol: end the method earlier when the norm of the residual is less than tol
         set_grad: if True set t.grad to the hypergradient for every t in hparams
         stochastic: set this to True when fp_map is not a deterministic function of its inputs

     Returns:
         the list of hypergradients for each element in hparams
     """
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(params, hparams)
            Jfp_mapTv = torch_grad(w_mapped_in, params, grad_outputs=xs, retain_graph=False)
        else:
            Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = CG_torch.cg(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def CG_normaleq(params: List[Tensor],
                hparams: List[Tensor],
                K: int ,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True) -> List[Tensor]:
    """ Similar to CG but the conjugate gradient is applied on the normal equation (has a higher time complexity)"""
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    w_mapped = fp_map(params, hparams)

    def dfp_map_dw(xs):
        Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        v_minus_Jfp_mapTv = [v - j for v, j in zip(xs, Jfp_mapTv)]

        # normal equation part
        Jfp_mapv_minus_Jfp_mapJfp_mapTv = jvp(lambda _params: fp_map(_params, hparams), params, v_minus_Jfp_mapTv)
        return [v - vv for v, vv in zip(v_minus_Jfp_mapTv, Jfp_mapv_minus_Jfp_mapJfp_mapTv)]

    v_minus_Jfp_mapv = [g - jfp_mapv for g, jfp_mapv in zip(grad_outer_w, jvp(
        lambda _params: fp_map(_params, hparams), params, grad_outer_w))]
    vs = CG_torch.cg(dfp_map_dw, v_minus_Jfp_mapv, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def neumann(params: List[Tensor],
            hparams: List[Tensor],
            K: int ,
            fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
            outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
            tol=1e-10,
            set_grad=True) -> List[Tensor]:
    """ Saves one iteration from the fixed point method"""

    # from https://arxiv.org/pdf/1803.06396.pdf,  should return the same gradient of fixed point K+1
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    w_mapped = fp_map(params, hparams)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for k in range(K):
        gs_prev_vec = gs_vec
        vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


def exact(opt_params_f: Callable[[List[Tensor]], List[Tensor]],
          hparams: List[Tensor],
          outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
          set_grad=True) -> List[Tensor]:
    """
    Computes the exact hypergradient using backpropagation and exploting the closed form torch differentiable function
    that computes the optimal parameters given the hyperparameters (opt_params_f).
    """
    grads = torch_grad(outer_loss(opt_params_f(hparams), hparams), hparams)
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


def stoch_AID(
    params: List[Tensor],
    hparams: List[Tensor],
    outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
    K: int,
    J_inner: int = 1,
    J_outer: int = 1,
    fp_map: Union[Callable[[List[Tensor], List[Tensor]], List[Tensor]], None] = None,
    inner_loss: Union[Callable[[List[Tensor], List[Tensor]], Tensor], None] = None,
    linsys_start: Union[List[Tensor], None] = None,
    stoch_outer: bool = False,
    stoch_inner: bool = False,
    optim_build: Union[Callable[..., Tuple[Optimizer, Any]], None] = None,
    opt_params: dict = None,
    set_grad: bool = True,
    verbose: bool = True,
):
    """
    Computes the hypergradient by solving the linear system by applying K steps of the optimizer output of the optim_build function,
    this should be a torch.optim.Optimizer. optim_build (optionally) returns also a scheduler whose step()
    method is called after every iteration of the optimizer.

    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        K: the number of iteration of the LS solver which is given as output of optim_build
        J_inner: the minibatch size used to compute the jacobian w.r.t. hparams of fp_map
        J_outer: the minibatch size used to compute the gradient w.r.t. params and hparams  of outer_loss
        fp_map: the fixed point map which defines the inner problem, used if inner_loss is None
        inner_loss: the loss of the inner problem, used if fp_map is None
        linsys_start: starting point of the linear system, set to the 0 vector if None
        stoch_outer: set to True if outer_loss is stochastic, otherwise set to False
        stoch_inner: set to True if fp_map or inner_loss is stochastic, otherwise False
        optim_build: function used to obtain the linear system optimizer
        opt_params: parameters of he linear system optimizer (input of optim_build)
        set_grad: if True set t.grad to the hypergradient for every t in hparams
        verbose: print the distance between two consecutive iterates for the linear system.
    Returns:
        the list of hypergradients for each element in hparams
    """

    assert stoch_inner or (J_inner == 1)
    assert stoch_outer or (J_outer == 1)

    params = [w.detach().clone().requires_grad_(True) for w in params]

    if fp_map is not None:
        w_update_f = fp_map

        def v_update(v, jtv, g):
            return v - jtv - g

    elif inner_loss is not None:
        # In this case  w_update_f is the negative gradient
        def w_update_f(params, hparams):
            return torch.autograd.grad(
                -inner_loss(params, hparams), params, create_graph=True
            )

        def v_update(v, jtv, g):
            return -jtv - g

    else:
        raise NotImplementedError("Either fp_map or inner loss should be not None")

    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(
        o_loss, params, hparams, retain_graph=False
    )
    if stoch_outer:
        for _ in range(J_outer - 1):
            o_loss = outer_loss(params, hparams)
            grad_outer_w_1, grad_outer_hparams_1 = get_outer_gradients(
                o_loss, params, hparams, retain_graph=False
            )
            for g, g1 in zip(grad_outer_w, grad_outer_w_1):
                g += g1
            for g, g1 in zip(grad_outer_hparams, grad_outer_hparams_1):
                g += g1

        for g in grad_outer_w:
            g /= J_outer
        for g in grad_outer_hparams:
            g /= J_outer

    if stoch_inner:

        def w_updated():
            return w_update_f(params, hparams)

    else:
        w_new = w_update_f(params, hparams)

        def w_updated():
            return w_new

    def compute_and_set_grads(vs):
        Jfp_mapTv = torch_grad(
            w_updated(), params, grad_outputs=vs, retain_graph=not stoch_inner
        )

        for v, jtv, g in zip(vs, Jfp_mapTv, grad_outer_w):
            v.grad = torch.zeros_like(v)
            v.grad += v_update(v, jtv, g)

    if linsys_start is not None:
        vparams = [l.detach().clone() for l in linsys_start]
    else:
        vparams = [gw.detach().clone() for gw in grad_outer_w]

    if optim_build is None:
        optim = torch.optim.SGD(vparams, lr=1.0)
        scheduler = None
    else:
        if opt_params is None:
            optim, scheduler = optim_build(vparams)
        else:
            optim, scheduler = optim_build(vparams, **opt_params)

    # Solve the linear system
    for i in range(K):
        vparams_prev = [v.detach().clone() for v in vparams]
        optim.zero_grad()
        compute_and_set_grads(vparams)
        optim.step()
        if scheduler:
            scheduler.step()
        if verbose and ((K < 5) or (i % (K // 5) == 0 or i == K - 1)):
            print(
                f"k={i}: linsys, ||v - v_prev|| = {[torch.norm(v - v_prev).item() for v, v_prev in zip(vparams, vparams_prev)]}"
            )

    if any(
        [(torch.isnan(torch.norm(v)) or torch.isinf(torch.norm(v))) for v in vparams]
    ):
        raise ValueError("Hypergradient's linear system diverged!")

    grads_indirect = [torch.zeros_like(g) for g in hparams]

    # Compute Jvp w.r.t lambda
    for i in range(J_inner):
        retain_graph = (not stoch_inner) and (i < J_inner - 1)

        djac_wrt_lambda = torch_grad(
            w_updated(),
            hparams,
            grad_outputs=vparams,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        for g, g1 in zip(grads_indirect, djac_wrt_lambda):
            if g1 is not None:
                g += g1 / J_inner

    grads = [g + v for g, v in zip(grad_outer_hparams, grads_indirect)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads, vparams


# UTILS

def grd(a, b):
    return torch.autograd.grad(a, b, create_graph=True, retain_graph=True)


def list_dot(l1, l2):  # extended dot product for lists
    return torch.stack([(a*b).sum() for a, b in zip(l1, l2)]).sum()


def jvp(fp_map, params, vs):
    dummy = [torch.ones_like(phw).requires_grad_(True) for phw in fp_map(params)]
    g1 = grd(list_dot(fp_map(params), dummy), params)
    return grd(list_dot(vs, g1), dummy)


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


