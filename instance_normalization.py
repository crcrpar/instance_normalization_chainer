import numpy

import chainer
from chainer.backends import cuda
from chainer import configuration
from chainer import functions
from chainer import links
from chainer.utils import argument


class InstanceNormalization(links.BatchNormalization):

    def __init__(self, size, decay=0.9, eps=1e-5, dtype=None,
                 use_gamma=False, use_beta=False,
                 initial_gamma=None, initial_beta=None,
                 track_runnign_stats=False):
        super(InstanceNormalization, self).__init__(
            size, decay, eps, dtype, use_gamma, use_beta,
            initial_gamma, initial_beta
        )

        self.track_runnign_stats = track_runnign_stats

    def forward(self, x, **kwargs):
        finetune, = argument.parse_kwargs(
            kwargs, ('finetune', False),
            test='test argument is not supported anymore. '
                 'Use chainer.using_config')

        if self.avg_mean is None:
            param_shape = tuple([
                d
                for i, d in enumerate(x.shape)
                if i not in self.axis])

        gamma = self.gamma
        if gamma is None:
            with cuda.get_device_from_id(self._device_id):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype)

        beta = self.beta
        if beta is None:
            with cuda.get_device_from_id(self._device_id):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype)

        # Reshape
        org_shape = x.shape
        x = functions.reshape(x, (1, org_shape[0] * org_shape[1], -1))
        gamma = functions.repeat(gamma, org_shape[0], axis=0)
        beta = functions.repeat(beta, org_shape[0], axis=0)

        if not configuration.config.train and self.track_runnign_stats:
            # Use running average statistics or fine-tuned statistics.
            mean = self.avg_mean
            var = self.avg_var
            ret = functions.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps, axis=self.axis)
        else:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = batch_normalization.batch_normalization(
                x, gamma, beta, eps=self.eps, running_mean=self.avg_mean,
                running_var=self.avg_var, decay=decay, axis=self.axis)
        ret = functions.reshape(ret, org_shape)
        return ret


if __name__ == '__main__':
    x = numpy.random.randn(2, 3, 16, 16).astype('f')
    norm = InstanceNormalization(3)
    try:
        y = norm(x)
    except Exception:
        print("Failed to normalize `x`.")
    else:
        assert all([a == b for a, b in zip(x.shape, y.shape)]), "Shape not match."
        print("Succeed to normalize `x`.")
        y = y.array
        print("mean(x - y) = {:.05f}".format((x - y).mean()))
