from itertools import repeat

from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import torch
import hypergrad as hg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized
import time

import torch.nn.functional as F


# Helper functions to deal with cuda
cuda = True and torch.cuda.is_available()

default_tensor_str = 'torch.cuda.FloatTensor' if cuda else 'torch.FloatTensor'

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
torch.set_default_tensor_type(default_tensor_str)
#torch.multiprocessing.set_start_method('forkserver')

def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()
def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy()


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


# load twentynews and preprocess
val_size = 0.5
X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True,
                                     #remove=('headers', 'footers', 'quotes')
                                     )
x_test, y_test = fetch_20newsgroups_vectorized(subset='test', return_X_y=True,
                                               #remove=('headers', 'footers', 'quotes')
                                               )


x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size)


train_samples, n_features = x_train.shape
test_samples, n_features = x_test.shape
val_samples, n_features = x_val.shape
n_classes = np.unique(y_train).shape[0]

print('Dataset 20newsgroup, train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i'
      % (train_samples, val_samples, test_samples, n_features, n_classes))


ys = [frnp(y_train).long(), frnp(y_val).long(), frnp(y_test).long()]
xs = [x_train, x_val, x_test]


def from_sparse(x):
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


if cuda:
    xs = [from_sparse(x).cuda() for x in xs]
else:
    xs = [from_sparse(x) for x in xs]

x_train, x_val, x_test = xs
y_train, y_val, y_test = ys


class CustomTensorIterator:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        self.loader = DataLoader(TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx


# torch.DataLoader has problems with sparse tensor on GPU
train_batch_size = len(y_train)
val_batch_size = len(y_val)

iterators = []
for bs, x, y in [(train_batch_size, x_train, y_train), (val_batch_size, x_val, y_val)]:
    if bs < len(y):
        print('making iterator with batch size ', bs)
        iterators.append(CustomTensorIterator([x, y], batch_size=bs, shuffle=True, **kwargs))
    else:
        iterators.append(repeat([x, y]))

train_iterator, val_iterator = iterators

# HPO set up
n_steps = 500
outer_lr, outer_mu = 100.0, 0.0  # nice with 100.0, 0.0 (torch.SGD) tested with T, K = 5, 10 and CG
inner_lr, inner_mu = 100., 0.9   # nice with 100., 0.9 (HeavyBall) tested with T, K = 5, 10 and CG
T, K = 10, 10
tol = 1e-12
warm_start = True
bias = False  # without bias outer_lr can be bigger (much faster convergence)

train_log_interval = 100
val_log_interval = 1

l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
#l2_reg_params = (-20.*torch.ones(1)).requires_grad_(True)  # one l2 hp only (best when really low)
l1_reg_params = (0.*torch.ones(1)).requires_grad_(True)  # one l1 hp only (best when really low)
#l1_reg_params = (-1.*torch.ones(n_features)).requires_grad_(True)

hparams = [l2_reg_params]

ones_dxc = torch.ones(n_features, n_classes)


def reg_f(params, l2_reg_params, l1_reg_params=None):
    r = 0.5 * ((params[0] ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
    if l1_reg_params is not None:
        r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
    return r


outer_opt = torch.optim.SGD(lr=outer_lr, momentum=outer_mu, params=hparams)
#outer_opt = torch.optim.Adam(lr=0.01, params=hparams)


params_history = []
val_losses, val_accs = [], []
test_losses, test_accs = [], []

w = torch.zeros(n_features, n_classes).requires_grad_(True)
parameters = [w]

if bias:
    b = torch.zeros(n_classes).requires_grad_(True)
    parameters.append(b)


def out_f(x, params):
    out = x @ params[0]
    out += params[1] if len(params) == 2 else 0
    return out


def train_loss(params, hparams, data):
    x_mb, y_mb = data
    out = out_f(x_mb,  params)
    return F.cross_entropy(out, y_mb) + reg_f(params, *hparams)


def val_loss(opt_params, hparams):
    x_mb, y_mb = next(val_iterator)
    out = out_f(x_mb,  opt_params[:len(parameters)])
    val_loss = F.cross_entropy(out, y_mb)
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

    val_losses.append(tonp(val_loss))
    val_accs.append(acc)
    return val_loss


def eval(params, x, y):
    out = out_f(x,  params)
    loss = F.cross_entropy(out, y)
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

    return loss, acc


if inner_mu > 0:
    #inner_opt = hg.Momentum(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
    inner_opt = hg.HeavyBall(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
else:
    inner_opt = hg.GradientDescent(train_loss, inner_lr, data_or_iter=train_iterator)

inner_opt_cg = hg.GradientDescent(train_loss, 1., data_or_iter=train_iterator)


params_history = []
total_time = 0
for o_step in range(n_steps):
    start_time = time.time()

    inner_losses = []
    params_history = [parameters]
    for t in range(T):
        params_history.append(inner_opt(params_history[-1], hparams, create_graph=False))
        inner_losses.append(inner_opt.curr_loss)

        if t % train_log_interval == 0 or t == T-1:
            print('t={} loss: {}'.format(t, inner_losses[-1]))

    final_params = params_history[-1]

    outer_opt.zero_grad()
    #hg.reverse(params_history[-K-1:], hparams, [inner_opt]*K, val_loss)
    #hg.fixed_point(final_params, hparams, K, inner_opt, val_loss, stochastic=False, tol=tol)
    hg.CG(final_params[:len(parameters)], hparams, K, inner_opt_cg, val_loss, stochastic=False, tol=tol)
    outer_opt.step()

    for p, new_p in zip(parameters, final_params[:len(parameters)]):
        if warm_start:
            p.data = new_p
        else:
            p.data = torch.zeros_like(p)

    iter_time = time.time() - start_time
    total_time += iter_time
    if o_step % val_log_interval == 0 or o_step == T-1:
        test_loss, test_acc = eval(final_params[:len(parameters)], x_test, y_test)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print('o_step={} ({:.2e}s) Val loss: {:.4e}, Val Acc: {:.2f}%'.format(o_step, iter_time, val_losses[-1],
                                                                              100*val_accs[-1]))
        print('          Test loss: {:.4e}, Test Acc: {:.2f}%'.format(test_loss, 100*test_acc))
        print('          l2_hp norm: {:.4e}'.format(torch.norm(hparams[0])))
        if len(hparams) == 2:
            print('          l1_hp : ', torch.norm(hparams[1]))

print('HPO ended in {:.2e} seconds\n'.format(total_time))

plt.title('val_accuracy')
plt.plot(val_accs)
plt.show()

plt.title('test_accuracy')
plt.plot(test_accs)
plt.show()

# Final Train on both train and validation sets
x_train_val = torch.cat([x_train, x_val], dim=0)
y_train_val = torch.cat([y_train, y_val], dim=0)
train_val_batch_size = len(x_train_val)


if train_val_batch_size < len(y_train_val):
    print('making iterator with batch size ', bs)
    train_val_iterator = CustomTensorIterator([x_train_val, y_train_val], batch_size=train_val_batch_size, shuffle=True, **kwargs)
else:
    train_val_iterator = repeat([x_train_val, y_train_val])

if inner_mu > 0:
    # inner_opt = hg.Momentum(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
    inner_opt = hg.HeavyBall(train_loss, inner_lr, inner_mu, data_or_iter=train_val_iterator)
else:
    inner_opt = hg.GradientDescent(train_loss, inner_lr, data_or_iter=train_val_iterator)


T_final = 4000
w = torch.zeros(n_features, n_classes).requires_grad_(True)
parameters = [w]

if bias:
    b = torch.zeros(n_classes).requires_grad_(True)
    parameters.append(b)

opt_params = inner_opt.get_opt_params(parameters)

print('Final training on both train and validation sets with final hyperparameters')
for t in range(T_final):
    opt_params = inner_opt(opt_params, hparams, create_graph=False)
    train_loss = inner_opt.curr_loss

    if t % train_log_interval == 0 or t == T_final-1:
        test_loss, test_acc = eval(opt_params[:len(parameters)], x_test, y_test)
        print('t={} final loss: {}'.format(t, train_loss))
        print('          Test loss: {}, Test Acc: {}'.format(test_loss, test_acc))

