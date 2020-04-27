from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import higher
from  torch.utils.checkpoint import checkpoint
from sklearn.model_selection import train_test_split

"""
Hypercleaning  on mnist with higher integration.

The 60000 training examples of the MNIST dataset are divided in 1000 (validation) and 5900 (training) and some percentage
of the training labels (e.g. 50%) are changed randomly. 

The CNN below achieves < 91%  test accuracy when trained on the corrupted training set + the validation set 
or on the validation only(to verify). By weighting the loss of each examples with an hyperparameter trained using
a bilevel scheme with warm-start you can easily reach 96/97% accuracy.

this experiment is similar to the one in 
Mehra, A., & Hamm, J. (2019). Penalty Method for Inversion-Free Deep Bilevel Optimization.
which is inspired by the simpler one in 
Franceschi, L., Donini, M., Frasconi, P., & Pontil, M. (2017). Forward and reverse gradient-based hyperparameter optimization.
"""



import hg


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


class CustomLoader:
    """
    Needed to deal with hyperparameters corresponding to each example and minibatches.
    Uses torch.utils.DataLoader on an array of indices.
    """
    def __init__(self, x, y, batch_size, **loader_kwargs):
        self.x = x
        self.y = y
        self.epoch, self.iter = 0, 0
        self.batch_size = batch_size

        self.loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(y))),
                                                  batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            self.epoch += 1
            self.iter = 0
            idx = next(self.iterator)

        self.iter += 1

        return [self.x[idx], self.y[idx], *[a[idx] for a in args]]


def train(hparams, model, fp_map, train_loader: CustomLoader, n_steps, log_interval):
    model.train()
    params_history = [model.fast_params]  # model should be a functional module from higher monkeypatch
    fp_map_history = []

    for t in range(n_steps):
        fp_map_history.append(fp_map)
        params_history.append(fp_map(params_history[-1], hparams))

        if t % log_interval == 0:
            print('t={}, epoch={} [{}/{}]\tLoss: {:.6f}'.format(
                t, train_loader.epoch, train_loader.iter * train_loader.batch_size,
                len(train_loader.y), fp_map.loss.item()))

    return params_history, fp_map_history


def eval_model(params, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, params=params)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Data HyperCleaner')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--val-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--flip-perc', type=float, default=0.5, metavar='M',
                        help='Percentage of flipped labels examples (default: 0.5)')
    parser.add_argument('--n_steps', type=int, default=10000, metavar='N',
                        help='number of outer optimization steps')
    parser.add_argument('--T', type=int, default=10, metavar='N',
                        help='number of inner steps to train')
    parser.add_argument('--K', type=int, default=10, metavar='N',
                        help='number of backward steps')
    parser.add_argument('--inner-lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: .1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--eval_interval', type=float, default=10, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Helper functions to deal with cuda and double precision
    cuda = not args.no_cuda and torch.cuda.is_available()
    double_precision = False

    def frnp(x):
        t = torch.from_numpy(x).cuda() if cuda else torch.from_numpy(x)
        return t if double_precision else t.float()

    def tonp(x, cuda=cuda):
        return x.detach().cpu().numpy() if cuda else x.detach().numpy()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # mnist_mean, mnist_std =0.1307, 0.3081

    mnist_train = datasets.MNIST('../data', download=True, train=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    val_size = 0.016666  # 1000 val examples
    #val_size = 0.166666  # 10^4 val example

    x = mnist_train.data.numpy()/255.
    y = mnist_train.targets.numpy()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

    x_train, x_val, y_train, y_val = frnp(x_train).unsqueeze(1), frnp(x_val).unsqueeze(1),\
                                     frnp(y_train).long(), frnp(y_val).long()

    #limit train set
    #x_train, y_train = x_train[:1000], y_train[:1000]

    #flip labels
    n_flip = int(args.flip_perc*len(y_train))
    y_train_oracle = y_train.clone()
    for i in range(n_flip):
        while y_train[i] == y_train_oracle[i]:
            y_train[i] = torch.randint(low=0, high=10, size=(1,))
    #y_train[:n_flip] = torch.LongTensor(n_flip).random_(0, 10)


    train_loader = CustomLoader(x_train, y_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = CustomLoader(x_val, y_val, batch_size=args.val_batch_size, shuffle=True, **kwargs)



    hparams = [torch.zeros_like(y_train).float().requires_grad_(True)]
    #outer_opt = optim.Adam(lr=args.lr, params=hparams)
    outer_opt = optim.SGD(lr=args.lr, momentum=0.9, params=hparams)

    model_nf = Net().to(device)

    for k in range(args.n_steps):
        #debug.print_tensors()

        model = higher.monkeypatch(model_nf, device=device, copy_initial_weights=True)
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        # train_losses = []
        val_losses = []
        val_accs = []

        class OPTMAP:
            def __init__(self):
                self.loss = None

            def __call__(self, params, hparams):
                x, y, exw = train_loader.__next__(hparams[0])
                self.loss = (torch.sigmoid(exw) * F.nll_loss(model(x, params=params), y, reduction='none')).mean()
                return hg.gd_step(params, self.loss, args.inner_lr, create_graph=True)

        fp_map = OPTMAP()

        def val_loss(params, hparams):
            x, y = next(val_loader)
            output = model(x, params=params)
            val_loss = F.nll_loss(output, y)
            val_losses.append(tonp(val_loss))
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
            val_accs.append(acc)
            return val_loss

        params_history, fp_map_history = train(hparams, model, fp_map, train_loader,
                                               n_steps=args.T, log_interval=1)

        outer_opt.zero_grad()
        hg.fixed_point(params_history[-1], hparams, K=args.K, fp_map=fp_map, outer_loss=val_loss, stochastic=False)
        #hg.CG(params_history[-1], hparams, K=args.K, fp_map=fp_map, outer_loss=val_loss, stochastic=True)
        #hg.reverse(params_history, hparams, K=args.K, fp_map_history=fp_map_history,outer_loss=val_loss)
        #hg.reverse_unroll(params_history[-1], hparams, outer_loss=val_loss)
        outer_opt.step()

        model_nf = Net().to(device)
        for p, up in zip(model_nf.parameters(), params_history[-1]):
            p.data = up.data

        if k % args.eval_interval == 0 or k == args.n_steps-1:
            print('\nk={}, val Loss, acc: {:.2e}, {:.2f}%'.format(k, val_losses[-1], 100.*val_accs[-1]))
            eval_model(params_history[-1], model, device, test_loader)

            if args.save_model:
                torch.save(model.state_dict(), "mnist_cnn_k{}.pt".format(k))


if __name__ == '__main__':
    main()