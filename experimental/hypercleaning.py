import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from  torch.utils.checkpoint import checkpoint
from sklearn.model_selection import train_test_split


import higher
import hg

"""
This experiment is similar to the one in 
Mehra, A., & Hamm, J. (2019). Penalty Method for Inversion-Free Deep Bilevel Optimization.
which is inspired by the simpler one in 
Franceschi, L., Donini, M., Frasconi, P., & Pontil, M. (2017). Forward and reverse gradient-based hyperparameter optimization.

it uses Higher (https://github.com/facebookresearch/higher) to get a stateless CNN.
Works with Higher version 0.1.5 (version 0.1.4 causes memory leaks!).

The setting is as follows.
The training examples of the MNIST dataset are divided in validation (1000 examples) and training (5900 examples). 
Some percentage of the training labels (e.g. 50%) are changed randomly (training set is corrupted). 

The training loss of each example is the classical cross-entropy weighted by an hyperparmaeter. 
The hyperparameters are optimized through a bilevel scheme where the outer loss is the cross entroy 
over the validation set. This procedure will be called hypercleaning because the validation set
is used to "clean" the corrupted training set.

SimpleCNN achieves 93% test accuracy only on validation (verified with Momentum)
                   96/97% test accuracy with validation + 50% corrupted train
                   96/97 test accuracy with validation + 50% corrupted train + hypercleaning

MehraCNN achieves 94/95% test accuracy only on validation (verified with Momentum)
                   ~98% test accuracy with validation + 50% corrupted train
                   ~98% test accuracy with validation + 50% corrupted train + hypercleaning

This differ greatly from Mehra et al 2019 which report < 91% accuracy for the settings without cleaning.
I supposed they wrongly reported the accuracy of the feed forward model in that table .
"""


def main():
    parser = argparse.ArgumentParser(description='Data HyperCleaner')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--val-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--val-perc', type=float, default=1- 0.0166666, metavar='M',
                        help='Percentage of examples in validation (default: 0.016666 = 1000 examples)')
    parser.add_argument('--flip-perc', type=float, default=0.0, metavar='M',
                        help='Percentage of flipped labels examples (default: 0.5)')

    parser.add_argument('--diff-train', action='store_true', default=False,
                        help='differentiable train loop when true')

    parser.add_argument('--n_steps', type=int, default=1000, metavar='N',
                        help='number of outer optimization steps')
    parser.add_argument('--no-warm_start', action='store_true', default=False,
                        help='disables warm-start on the network parameters')
    parser.add_argument('--T', type=int, default=10, metavar='N',
                        help='number of inner training steps')
    parser.add_argument('--K', type=int, default=10, metavar='N',
                        help='number of backward steps')
    parser.add_argument('--inner-lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: .1)')
    parser.add_argument('--lr', type=float, default=0.0, metavar='M',
                        help='Learning rate step (default: 0.1)')

    parser.add_argument('--eval_interval', type=float, default=10, metavar='M',
                        help='test set evaluation interval (default: 10)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--inner-log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()
    def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy()

    torch.manual_seed(args.seed)

    mnist_train = datasets.MNIST('../data', download=True, train=True)
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    x = mnist_train.data.numpy()/255.
    y = mnist_train.targets.numpy()
    del mnist_train

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.val_perc)

    x_train, x_val, y_train, y_val = frnp(x_train).unsqueeze(1), frnp(x_val).unsqueeze(1),\
                                     frnp(y_train).long(), frnp(y_val).long()

    # x_train, y_train = x_train[:1000], y_train[:1000] # limit train set (DEBUG)

    # flip training labels
    n_flip = int(args.flip_perc*len(y_train))
    y_train_oracle = y_train.clone()
    for i in range(n_flip):
        while y_train[i] == y_train_oracle[i]:
            y_train.data[i] = torch.randint(low=0, high=10, size=(1,))

    train_iterator = CustomDataIterator(x_train, y_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_iterator = CustomDataIterator(x_val, y_val, batch_size=args.val_batch_size, shuffle=True, **kwargs)

    loss_weights = torch.zeros_like(y_train).float().requires_grad_(True)

    #outer_opt = optim.Adam(lr=args.lr, params=hparams)
    outer_opt = torch.optim.SGD(lr=args.lr, momentum=0.9, params=[loss_weights])

    model = MehraCNN().to(device)

    for k in range(args.n_steps):
        start_time = time.time()
        fmodel = higher.monkeypatch(model, device=device, copy_initial_weights=True)

        class GDMap:
            def __init__(self):
                self.loss = None

            def __call__(self, params, hparams):
                x, y, exw = train_iterator.__next__(hparams[0])
                self.loss = (torch.sigmoid(exw) * F.nll_loss(fmodel(x, params=params), y, reduction='none')).mean()
                return hg.gd_step(params, self.loss, args.inner_lr, create_graph=True)

        gd_map = GDMap()
        val_losses, val_accs = [], []

        def val_loss(params, hparams):
            data, targets = next(val_iterator)
            output = fmodel(data, params=params)
            val_loss = F.nll_loss(output, targets)
            val_losses.append(tonp(val_loss))
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(targets.view_as(pred)).sum().item() / len(targets)
            val_accs.append(acc)
            return val_loss

        if args.diff_train:
            params_history, fp_map_history = diff_train([loss_weights], fmodel, gd_map, train_iterator,
                                                        n_steps=args.T, log_interval=args.inner_log_interval)
        else:
            optim = torch.optim.SGD(lr=args.inner_lr, momentum=0.9, params=model.parameters())
            #optim = torch.optim.Adadelta(lr=args.inner_lr, params=model.parameters())
            params_history = train([loss_weights], model, optim, train_iterator, n_steps=args.T,
                                   log_interval=args.inner_log_interval, trajectory=True)
            fp_map_history = [gd_map]*args.T

        outer_opt.zero_grad()
        # comment out the hypergradient approximation method below that you wish to use
        hg.fixed_point(params_history[-1], [loss_weights], K=args.K, fp_map=gd_map, outer_loss=val_loss,
                       stochastic=False)
        #hg.CG(params_history[-1], hparams, K=args.K, fp_map=fp_map, outer_loss=val_loss, stochastic=True)
        #hg.reverse(params_history, hparams, K=args.K, fp_map_history=fp_map_history,outer_loss=val_loss)
        #hg.reverse_unroll(params_history[-1], hparams, outer_loss=val_loss)
        outer_opt.step()

        if not args.no_warm_start:
            for p, up in zip(model.parameters(), params_history[-1]):
                p.data = up.data

        step_time = time.time() - start_time
        if k % args.eval_interval == 0 or k == args.n_steps-1:
            print('\nouter step={} ({:.2e}s)'.format(k, step_time))
            print('Val Set: Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_losses[-1], 100. * val_accs[-1]))

            corr = torch.ones_like(y_train)
            corr[:n_flip] = 0
            acc_hp = corr.eq(torch.sigmoid(loss_weights) > 0.5).float().sum() / len(corr)
            print('Corr examples Accuracy: {:.2f}%'.format(100.*acc_hp))

        if k % args.eval_interval == 0 or k == args.n_steps-1:
            eval_model(params_history[-1], fmodel, device, test_loader)
            if args.save_model:
                torch.save(fmodel.state_dict(), "mnist_cnn_k{}.pt".format(k))
        if k % args.eval_interval == 0 or k == args.n_steps-1:
            print('\n')


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MehraCNN(nn.Module):
    """ same CNN as Mehra et al 2019 (from https://github.com/jihunhamm/bilevel-penalty/)"""

    def __init__(self):
        super(MehraCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.dropout4 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class CustomDataIterator:
    """
    Needed to deal with hyperparameters corresponding to each example and minibatches.
    Uses torch.utils.DataLoader on an array of indices.
    """
    def __init__(self, x, y, batch_size, **loader_kwargs):
        self.x = x
        self.y = y
        self.epoch, self.step = 0, 0
        self.batch_size = batch_size

        # loader on the array of indices
        self.loader = DataLoader(TensorDataset(torch.arange(len(y))), batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            self.epoch += 1
            self.step = 0
            idx = next(self.iterator)

        self.step += 1

        return [self.x[idx], self.y[idx], *[a[idx] for a in args]]


def train(hparams, model, optim, data_iterator: CustomDataIterator, n_steps, log_interval,
          trajectory=False):

    params_history = []
    for t in range(n_steps):
        if trajectory:
            params_history.append([p.detach().clone().requires_grad_(True) for p in model.parameters()])

        x, y, exw = data_iterator.__next__(hparams[0])
        loss = (torch.sigmoid(exw) * F.nll_loss(model(x), y, reduction='none')).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        if t % log_interval == 0 or t == n_steps-1:
            print('t={}, epoch={}, mb={} [{}/{}] Loss: {:.6f}'.format(
                t, data_iterator.epoch, data_iterator.step, data_iterator.step * data_iterator.batch_size,
                len(data_iterator.y), loss.item()))

    params_history.append([p.detach().clone().requires_grad_(True) for p in model.parameters()])
    return params_history


def diff_train(hparams, params, opt_step, data_iterator: CustomDataIterator, n_steps, log_interval):
    params_history = [params]
    fp_map_history = []

    for t in range(n_steps):
        fp_map_history.append(opt_step)
        params_history.append(opt_step(params_history[-1], hparams))

        if t % log_interval == 0 or t == n_steps-1:
            print('t={}, epoch={}, mb={} [{}/{}] Loss: {:.6f}'.format(
                t, data_iterator.epoch, data_iterator.step, data_iterator.step * data_iterator.batch_size,
                len(data_iterator.y), opt_step.loss.item()))

    return params_history, fp_map_history


def eval_model(params, model, device, data_loader: DataLoader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, params=params)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('Test set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    main()