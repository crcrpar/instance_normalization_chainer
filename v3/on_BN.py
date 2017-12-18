import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import functions
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable


class InstanceNormalization(link.Link):

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 valid_test=False, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(InstanceNormalization, self).__init__()
        self.valid_test = valid_test
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.N = 0
        self.register_persistent('avg_mean')
        self.register_persistent('avg_var')
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluation during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', False)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        # check argument
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        # reshape input x
        original_shape = x.shape
        batch_size, n_ch = original_shape[:2]
        new_shape = (1, batch_size * n_ch) + original_shape[2:]
        reshaped_x = functions.reshape(x, new_shape)

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        mean = chainer.as_variable(self.xp.hstack([self.avg_mean] * batch_size))
        var = chainer.as_variable(self.xp.hstack([self.avg_var] * batch_size))
        gamma = chainer.as_variable(self.xp.hstack([gamma.array] * batch_size))
        beta = chainer.as_variable(self.xp.hstack([beta.array] * batch_size))
        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = functions.batch_normalization(
                reshaped_x, gamma, beta, eps=self.eps, running_mean=mean,
                running_var=var, decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            ret = functions.fixed_batch_normalization(
                reshaped_x, gamma, beta, mean, var, self.eps)

        # ret is normalized input x
        return functions.reshape(ret, original_shape)


if __name__ == '__main__':
    import numpy as np
    base_shape = [10, 3]
    with chainer.using_config('debug', True):
        for i, n_element in enumerate([32, 32, 32]):
            base_shape.append(n_element)
            print('# {} th: input shape: {}'.format(i, base_shape))
            x_array = np.random.normal(size=base_shape).astype(np.float32)
            x = chainer.as_variable(x_array)
            layer = InstanceNormalization(base_shape[1])
            y = layer(x)
            # calculate y_hat manually
            axes = tuple(range(2, len(base_shape)))
            x_mean = np.mean(x_array, axis=axes, keepdims=True)
            x_var = np.var(x_array, axis=axes, keepdims=True) + 1e-5
            x_std = np.sqrt(x_var)
            y_hat = (x_array - x_mean) / x_std
            diff = y.array - y_hat
            print('*** diff ***')
            print('\tmean: {:03f},\n\tstd: {:.03f}'.format(
                np.mean(diff), np.std(diff)))

        base_shape = [10, 3]
        with chainer.using_config('train', False):
            print('\n# test mode\n')
            for i, n_element in enumerate([32, 32, 32]):
                base_shape.append(n_element)
                print('# {} th: input shape: {}'.format(i, base_shape))
                x_array = np.random.normal(size=base_shape).astype(np.float32)
                x = chainer.as_variable(x_array)
                layer = InstanceNormalization(base_shape[1])
                y = layer(x)
                axes = tuple(range(2, len(base_shape)))
                x_mean = np.mean(x_array, axis=axes, keepdims=True)
                x_var = np.var(x_array, axis=axes, keepdims=True) + 1e-5
                x_std = np.sqrt(x_var)
                y_hat = (x_array - x_mean) / x_std
                diff = y.array - y_hat
                print('*** diff ***')
                print('\tmean: {:03f},\n\tstd: {:.03f}'.format(
                    np.mean(diff), np.std(diff)))


"""
○ → python instance_norm.py
# 0 th: input shape: [10, 3, 32]
*** diff ***
        mean: -0.000000,
        std: 0.000
# 1 th: input shape: [10, 3, 32, 32]
*** diff ***
        mean: -0.000000,
        std: 0.000
# 2 th: input shape: [10, 3, 32, 32, 32]
*** diff ***
        mean: -0.000000,
        std: 0.000
        
# test mode
# 0 th: input shape: [10, 3, 32]
*** diff ***
        mean: 14.126040,
        std: 227.823
# 1 th: input shape: [10, 3, 32, 32]
*** diff ***
        mean: -0.286635,
        std: 221.926
# 2 th: input shape: [10, 3, 32, 32, 32]
*** diff ***
        mean: -0.064297,
        std: 222.492
"""
