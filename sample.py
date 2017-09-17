import matplotlib
matplotlib.use('Agg')

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from instance_normalization import InstanceNormalization


class ShallowConv(chainer.Chain):

    """Shallow Conv

    This is a shallow convolutional network to check whether
    InstanceNormalization work or not.
    """

    def __init__(self):
        super(ShallowConv, self).__init__()
        with self.init_scope():
            self.c_1 = L.Convolution2D(1, 3, 7, 2, 3)
            self.i_1 = InstanceNormalization(3)
            self.c_2 = L.Convolution2D(3, 6, 7, 4, 4)
            self.i_2 = InstanceNormalization(6)
            self.l_1 = L.Linear(None, 10)

    def __call__(self, x):
        h = F.relu(self.i_1(self.c_1(x)))
        h = F.relu(self.i_2(self.c_2(h)))
        bs = len(h)
        h = F.reshape(h, (bs, -1))
        return self.l_1(h)


def main(gpu_id=-1, bs=32, epoch=10, out='./result', resume=''):
    net = ShallowConv()
    model = L.Classifier(net)
    if gpu_id >= 0:
        chainer.cuda.get_device_from_id(gpu_id)
        model.to_gpu()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, bs)
    test_iter = chainer.iterators.SerialIterator(
        test, bs, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
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
