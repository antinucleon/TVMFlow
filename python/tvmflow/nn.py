from __future__ import absolute_import as _abs

from ._base import Variable
from ._nnvm import symbol as sym

float32 = 0


def relu(x, name):
    return sym.relu(x, name=name)


def conv2d(x, out_filter, in_filter, kernel, stride, pad, name):
    w = Variable(
        init=sym.normal(
            shape=[out_filter, in_filter, kernel, kernel], dtype=float32),
        name="%s_weight" % name)
    #b = Variable(
    #    init=sym.zeros(shape=[1, out_filter, 1, 1], dtype=float32),
    #    name="%s_bias" % name)
    out = sym.conv2d(x, w, pad=pad, stride=stride, name=name)
    #return sym.add_bias4d(out, b, name="%s_add_bias" % name)
    return out


def fully_connected(x, out_dim, in_dim, name):
    w = Variable(
        init=sym.normal(
            shape=[in_dim, out_dim], dtype=float32, scale=1.0 / out_dim),
        name="%s_weight" % name)
    b = Variable(
        init=sym.zeros(shape=[1, out_dim], dtype=float32),
        name="%s_bias" % name)
    out = sym.matmul(x, w, name=name)
    return sym.add_bias2d(out, b, name="%s_add_bias" % name)


def batch_norm(x, in_dim, name):
    gamma = Variable(
        init=sym.ones(shape=[1, in_dim, 1, 1], dtype=float32),
        name="%s_gamma" % name)
    beta = Variable(
        init=sym.zeros(shape=[1, in_dim, 1, 1], dtype=float32),
        name="%s_beta" % name)
    return sym.batch_norm(x, gamma, beta, name=name)


def softmax(x, name):
    return sym.softmax(x, name=name)


def global_pool(x, name):
    return sym.global_pool(x, name=name)


def flatten(x, name):
    return sym.flatten(x, name=name)


def logloss(y, label):
    return sym.reduce_mean(
        -sym.reduce_sum(label * sym.log(y), reduction_indices=[1]),
        keepdims=True)
