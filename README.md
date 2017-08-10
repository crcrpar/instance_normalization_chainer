# Instance Normalization
This is a [Chainer v2](https://chainer.org) implementation of Instance Normalization.
Note that this implementation will not work if your Chainer version is under 2.0.
Instance normalization is regarded as more suitable for `style transfer` task than batch normalization.
In Instance normalization, you normalize each mini batch using mean and variance of each tensor in one mini batch.
So the shapes of mean and variance should be `(batch_size, n_channel)`.

The original paper is found [here](http://arxiv.org/abs/1607.08022).
And my working history is found [here](https://github.com/crcrpar/chainer/tree/instance_norm).
