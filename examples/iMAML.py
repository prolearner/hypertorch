"""
Meta-learning Omniglot and mini-imagenet experiments with iMAML-GD (see [1] for more details).

The code is quite simple and easy to read thanks to the following two libraries which need both to be installed.
- higher: https://github.com/facebookresearch/higher (used to get stateless version of torch nn.Module-s)
- torchmeta: https://github.com/tristandeleu/pytorch-meta (used for meta-dataset loading and minibatching)


[1] Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019).
    Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems (pp. 113-124).
    https://arxiv.org/abs/1909.04630
"""

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

import higher

import hypergrad as hg


class Task:
    """
    Handles the train and valdation loss for a single task
    """
    def __init__(self, reg_param, meta_model, data, batch_size=None):
        device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmodel = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)

        self.n_params = len(list(meta_model.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def bias_reg_f(self, bias, params):
        # l2 biased regularization
        return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

    def train_loss_f(self, params, hparams):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        out = self.fmodel(self.train_input, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.bias_reg_f(hparams, params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        out = self.fmodel(self.test_input, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss


def main():
    parser = argparse.ArgumentParser(description='Data HyperCleaner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='N', help='omniglot or miniimagenet')
    parser.add_argument('--hg-mode', type=str, default='CG', metavar='N',
                        help='hypergradient approximation: CG or fixed_point')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    print(args, '\n')

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    log_interval = 1
    eval_interval = 500
    inner_log_interval = None
    inner_log_interval_test = None
    ways = 5
    batch_size = 16
    n_tasks_test = 1000  # usually 1000 tasks are used for testing
    T, K = 10, 5  # mini T=10, omni T=16
    T_test = T
    inner_lr = .1
    reg_param = 0.5  # mini 0.5, omni 2.0

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'omniglot':
        dataset = omniglot("data", ways=ways, shots=1, test_shots=15, meta_train=True, download=True)
        test_dataset = omniglot("data", ways=ways, shots=1, test_shots=15, meta_test=True, download=True)

        meta_model = get_cnn_omniglot(64, ways).to(device)
    elif args.dataset == 'miniimagenet':
        dataset = miniimagenet("data", ways=ways, shots=1, test_shots=15, meta_train=True, download=True)
        test_dataset = miniimagenet("data", ways=ways, shots=1, test_shots=15, meta_test=True, download=True)

        meta_model = get_cnn_miniimagenet(32, ways).to(device)
    else:
        raise NotImplementedError("DATASET NOT IMPLEMENTED! only omniglot and miniimagenet ")

    dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=batch_size, **kwargs)

    outer_opt = torch.optim.Adam(params=meta_model.parameters())
    # outer_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
    inner_opt_class = hg.GradientDescent
    inner_opt_kwargs = {'step_size': inner_lr}

    def get_inner_opt(train_loss):
        return inner_opt_class(train_loss, **inner_opt_kwargs)

    for k, batch in enumerate(dataloader):
        start_time = time.time()
        meta_model.train()

        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)

        outer_opt.zero_grad()

        val_loss, val_acc = 0, 0
        forward_time, backward_time = 0, 0
        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
            start_time_task = time.time()

            # single task set up
            task = Task(reg_param, meta_model,  (tr_x, tr_y, tst_x, tst_y), batch_size=tr_xs.shape[0])
            inner_opt = get_inner_opt(task.train_loss_f)

            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
            last_param = inner_loop(meta_model.parameters(), params, inner_opt, T, log_interval=inner_log_interval)[-1]
            forward_time_task = time.time() - start_time_task

            # single task hypergradient computation
            if args.hg_mode == 'CG':
                # This is the approximation used in the paper CG stands for conjugate gradient
                cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=1.)
                hg.CG(last_param, list(meta_model.parameters()), K=K, fp_map=cg_fp_map, outer_loss=task.val_loss_f)
            elif args.hg_mode == 'fixed_point':
                hg.fixed_point(last_param, list(meta_model.parameters()), K=K, fp_map=inner_opt,
                               outer_loss=task.val_loss_f)

            backward_time_task = time.time() - start_time_task - forward_time_task

            val_loss += task.val_loss
            val_acc += task.val_acc/task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task

        outer_opt.step()
        step_time = time.time() - start_time

        if k % log_interval == 0:
            print('MT k={} ({:.3f}s F: {:.3f}s, B: {:.3f}s) Val Loss: {:.2e}, Val Acc: {:.2f}.'
                  .format(k, step_time, forward_time, backward_time, val_loss, 100. * val_acc))

        if k % eval_interval == 0:
            test_losses, test_accs = evaluate(n_tasks_test, test_dataloader, meta_model, T_test, get_inner_opt,
                                          reg_param, log_interval=inner_log_interval_test)

            print("Test loss {:.2e} +- {:.2e}: Test acc: {:.2f} +- {:.2e} (mean +- std over {} tasks)."
                  .format(test_losses.mean(), test_losses.std(), 100. * test_accs.mean(),
                          100.*test_accs.std(), len(test_losses)))


def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(params_history[-1], hparams, create_graph=create_graph))

        if log_interval and (t % log_interval == 0 or t == n_steps-1):
            print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

    return params_history


def evaluate(n_tasks, dataloader, meta_model, n_steps, get_inner_opt, reg_param, log_interval=None):
    meta_model.train()
    device = next(meta_model.parameters()).device

    val_losses, val_accs = [], []
    for k, batch in enumerate(dataloader):
        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)

        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
            task = Task(reg_param, meta_model, (tr_x, tr_y, tst_x, tst_y))
            inner_opt = get_inner_opt(task.train_loss_f)

            params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
            last_param = inner_loop(meta_model.parameters(), params, inner_opt, n_steps, log_interval=log_interval)[-1]

            task.val_loss_f(last_param, meta_model.parameters())

            val_losses.append(task.val_loss)
            val_accs.append(task.val_acc)

            if len(val_accs) >= n_tasks:
                return np.array(val_losses), np.array(val_accs)


def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    return nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size, n_classes)
    )


def get_cnn_miniimagenet(hidden_size, n_classes):
    def conv_layer(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    return nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size*5*5, n_classes)
    )


if __name__ == '__main__':
    main()