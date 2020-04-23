import torch
from torch.autograd import grad as torch_grad
from hg import CG_torch
import traceback


# noinspection PyUnusedLocal
def reverse_unroll(ws, lmbd, outer_loss, set_grad=True):
    o_loss = outer_loss(ws, lmbd)
    grads = torch.autograd.grad(o_loss, lmbd, retain_graph=True)
    if set_grad:
        update_tensor_grads(lmbd, grads)
    return grads


# noinspection PyUnusedLocal
def reverse(ws_t, lmbd, K, phi_t, outer_loss, tol=1e-10, set_grad=True):
    ws_t = [[w.detach().requires_grad_(True) for w in ws] for ws in ws_t]
    o_loss = outer_loss(ws_t[-1], lmbd)
    grad_outer_w, grad_outer_lmbd = get_outer_gradients(o_loss, ws_t[-1], lmbd)

    alphas = grad_outer_w
    grads = [torch.zeros_like(w) for w in lmbd]
    K = min(K, len(ws_t) - 1)
    for k in range(-2, -(K + 2), -1):
        w_mapped = phi_t[k + 1](ws_t[k], lmbd)
        bs = grad_unused_zero(w_mapped, lmbd, grad_outputs=alphas, retain_graph=True)
        grads = [g + b for g, b in zip(grads, bs)]
        alphas = torch_grad(w_mapped, ws_t[k], grad_outputs=alphas)

    grads = [g + v for g, v in zip(grads, grad_outer_lmbd)]
    if set_grad:
        update_tensor_grads(lmbd, grads)

    return grads


def fixed_point(ws, lmbd, K, phi, outer_loss, tol=1e-10, set_grad=True):
    ws = [w.detach().requires_grad_(True) for w in ws]
    o_loss = outer_loss(ws, lmbd)
    grad_outer_w, grad_outer_lmbd = get_outer_gradients(o_loss, ws, lmbd)

    w_mapped = phi(ws, lmbd)
    vs = [torch.zeros_like(w) for w in ws]
    vs_vec = cat_list_to_tensor(vs)
    for k in range(K):
        vs_prev_vec = vs_vec
        vs = torch_grad(w_mapped, ws, grad_outputs=vs, retain_graph=True)
        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)
        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, lmbd, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_lmbd)]

    if set_grad:
        update_tensor_grads(lmbd, grads)

    return grads


def CG(ws, lmbd, K, phi, outer_loss, tol=1e-10, set_grad=True):
    ws = [w.detach().requires_grad_(True) for w in ws]
    o_loss = outer_loss(ws, lmbd)
    grad_outer_w, grad_outer_lmbd = get_outer_gradients(o_loss, ws, lmbd)

    w_mapped = phi(ws, lmbd)

    def dphi_dw(xs):
        JphiTv = torch_grad(w_mapped, ws, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, JphiTv)]

    vs = CG_torch.cg(dphi_dw, grad_outer_w, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    grads = torch_grad(w_mapped, lmbd, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_lmbd)]

    if set_grad:
        update_tensor_grads(lmbd, grads)

    return grads


def CG_normaleq(ws, lmbd, K, phi, outer_loss, tol=1e-10, set_grad=True):
    ws = [w.detach().requires_grad_(True) for w in ws]
    o_loss = outer_loss(ws, lmbd)
    grad_outer_w, grad_outer_lmbd = get_outer_gradients(o_loss, ws, lmbd)

    w_mapped = phi(ws, lmbd)

    def dphi_dw(xs):
        JphiTv = torch_grad(w_mapped, ws, grad_outputs=xs, retain_graph=True)
        v_minus_JphiTv = [v - j for v, j in zip(xs, JphiTv)]

        # normal equation part
        Jphiv_minus_JphiJphiTv = jvp(lambda _ws: phi(_ws, lmbd), ws, v_minus_JphiTv)
        return [v - vv for v, vv in zip(v_minus_JphiTv, Jphiv_minus_JphiJphiTv)]

    v_minus_Jphiv = [g - jphiv for g, jphiv in zip(grad_outer_w, jvp(
        lambda _ws: phi(_ws, lmbd), ws, grad_outer_w))]
    vs = CG_torch.cg(dphi_dw, v_minus_Jphiv, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    grads = torch_grad(w_mapped, lmbd, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_lmbd)]

    if set_grad:
        update_tensor_grads(lmbd, grads)

    return grads


def neumann(ws, lmbd, K, phi, outer_loss, tol=1e-10, set_grad=True):
    # from https://arxiv.org/pdf/1803.06396.pdf,  should return the same gradient of fixed point K+1
    ws = [w.detach().requires_grad_(True) for w in ws]
    o_loss = outer_loss(ws, lmbd)
    grad_outer_w, grad_outer_lmbd = get_outer_gradients(o_loss, ws, lmbd)

    w_mapped = phi(ws, lmbd)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for k in range(K):
        gs_prev_vec = gs_vec
        vs = torch_grad(w_mapped, ws, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, lmbd, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_lmbd)]
    if set_grad:
        update_tensor_grads(lmbd, grads)
    return grads


def exact(w_opt_f, lmbd, outer_loss, set_grad=True):
    grads = torch_grad(outer_loss(w_opt_f(lmbd), lmbd), lmbd)
    if set_grad:
        update_tensor_grads(lmbd, grads)
    return grads


# UTILS

def grd(a, b):
    return torch.autograd.grad(a, b, create_graph=True, retain_graph=True)


def list_dot(l1, l2):  # extended dot product for lists
    return torch.stack([(a*b).sum() for a, b in zip(l1, l2)]).sum()


def jvp(phi, ws, vs):
    dummy = [torch.ones_like(phw).requires_grad_(True) for phw in phi(ws)]
    g1 = grd(list_dot(phi(ws), dummy), ws)
    return grd(list_dot(vs, g1), dummy)


def get_outer_gradients(outer_loss, ws, lmbd, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, ws, retain_graph=retain_graph)
    grad_outer_lmbd = grad_unused_zero(outer_loss, lmbd, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_lmbd


def get_grad_func_fixed_args(grad_f, fixed_k=None, fixed_phi=None):
    # noinspection PyUnusedLocal
    def grad_func_fixed_args(w, lmbd, K, phi, outer_loss, set_grad=True):
        K = K if fixed_k is None else fixed_k
        phi = phi if fixed_phi is None else fixed_phi
        return grad_f(w, lmbd, K, phi, outer_loss, set_grad=True)

    return grad_func_fixed_args


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def update_tensor_grads(lmbd, grads):
    for l, g in zip(lmbd, grads):
        l.grad += g


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    try:
        grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                    retain_graph=retain_graph, create_graph=create_graph)
    except Exception as err:
        # TODO: remove this catch
        print('-----------------grad_unused_zero Exception!--------------------')
        traceback.print_tb(err.__traceback__)
        grads = [None] * len(inputs)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


