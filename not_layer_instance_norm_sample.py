"""Note that this program is possible only v3"""
import matplotlib
matplotlib.use('Agg')

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


def prepare_beta(size, init=0, dtype=np.float32):
    initial_beta = chainer.initializers._get_initializer(init)
    initial_beta.dtype = dtype
    beta = chainer.variable.Parameter(init, size)
    return beta


def prepare_gamma(size, init=1, dtype=np.float32):
    initial_gamma = chainer.initializers._get_initializer(init)
    initial_gamma.dtype = dtype
    gamma = chainer.variable.Parameter(init, size)
    return gamma


class ShallowConv(chainer.Chain):

    def __init__(self):
        super(ShallowConv, self).__init__()
        with self.init_scope():
            self.c_1 = L.Convolution2D(1, 3, 7, 2, 3, nobias=False)
            self.c_2 = L.Convolution2D(3, 6, 7, 4, 3, nobias=False)
            self.prob = L.Linear(None, 10)
            self.gamma_1 = prepare_gamma(3)
            self.beta_1 = prepare_beta(3)
            self.gamma_2 = prepare_gamma(6)
            self.beta_2 = prepare_beta(6)

    def __call__(self, x):
        h = F.relu(self.instance_norm(self.c_1(x), self.gamma_1, self.beta_1))
        h = F.relu(self.instance_norm(self.c_2(h), self.gamma_2, self.beta_2))
        bs = len(x)
        h = F.reshape(h, (bs, -1))
        return self.prob(h)

    def instance_norm(self, x, gamma=None, beta=None):
        mean = F.mean(x, axis=-1)
        mean = F.mean(mean, axis=-1)
        mean = F.broadcast_to(mean[Ellipsis, None, None], x.shape)
        var = F.squared_difference(x, mean)
        std = F.sqrt(var + 1e-5)
        x_hat = (x - mean) / std
        if gamma is not None:
            gamma = F.broadcast_to(gamma[None, Ellipsis, None, None], x.shape)
            beta = F.broadcast_to(beta[None, Ellipsis, None, None], x.shape)
            return gamma * x_hat + beta
        else:
            return x_hat


def main(gpu_id=-1, bs=32, epoch=20, out='./not_layer_result', resume=''):
    net = ShallowConv()
    model = L.Classifier(net)
    if gpu_id >= 0:
        chainer.cuda.get_device_from_id(gpu_id)
        model.to_gpu()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, bs)
    test_iter = chainer.iterators.SerialIterator(test, bs, repeat=False,
                                                 shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(extensions.ParameterStatistics(model.predictor))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
