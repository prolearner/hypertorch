import torch


class InnerOpt:
    def __init__(self, loss_f):
        self.loss_f = loss_f
        self.current_loss = None
        self.dim_mult = 0

    def step(self, wv, lmbd, create_graph):
        raise NotImplementedError

    def __call__(self, wv, lmbd, create_graph=True):
        with torch.enable_grad():
            return self.step(wv, lmbd, create_graph)

    def get_loss(self, ws, lmbd):
        self.current_loss = self.loss_f(ws, lmbd)
        return self.current_loss


class HeavyBall(InnerOpt):
    def __init__(self, loss_f, step_size_f, momentum_f):
        super(HeavyBall, self).__init__(loss_f)
        self.loss_f = loss_f
        self.step_size_f = step_size_f
        self.momentum_f = momentum_f
        self.dim_mult = 2

    def step(self, wv, lmbd, create_graph):
        n = len(wv) // 2
        ws, vs = wv[:n], wv[n:]
        loss = self.get_loss(ws, lmbd)
        sz, mu = self.step_size_f(lmbd), self.momentum_f(lmbd)
        w, v = heavy_ball_step(ws, vs, loss, sz,  mu, create_graph=create_graph)
        return [*w, *v]


class GradientDescent(InnerOpt):
    def __init__(self, loss_f, step_size_f):
        super(GradientDescent, self).__init__(loss_f)
        self.step_size_f = step_size_f
        self.dim_mult = 1

    def step(self, wv, lmbd, create_graph):
        loss = self.get_loss(wv, lmbd)
        sz = self.step_size_f(lmbd)
        return gd_step(wv, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g for w, g in zip(params, grads)]


def heavy_ball_step(params, vs, loss, step_size, momentum, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g + momentum * (w - v) for g, w, v in zip(grads, params, vs)], params


