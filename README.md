# Instance Normalization
This is a [Chainer v5](https://chainer.org) compatible implementation of Instance Normalization.

Instance normalization is regarded as more suitable for `style transfer` task than batch normalization qualitatively.
In Instance normalization, you normalize each mini batch using mean and variance of each tensor in one mini batch.
So the shapes of mean and variance should be `(batch_size, n_channel)`.

The original paper is found [here](http://arxiv.org/abs/1607.08022).

I'm looking forward to your review and/or correction.

# Comparison
This is comparison of InstanceNormalization layer and combination of `chainer.functions` and `chainer.variable.Parameter`.
![Comparison](https://raw.githubusercontent.com/crcrpar/instance_normalization_chainer/master/comparison.png)
