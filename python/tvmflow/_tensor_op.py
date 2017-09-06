"""Wrapping of certain ops for positional arguments.
Mainly because NNVM accepts kwargs for some additional arguments,
while TF sometimes support positional ops.
"""
from __future__ import absolute_import as _abs
from ._nnvm import symbol
from ._nnvm import _symbol_internal

from symbol import *
import numpy as np
import tvm
import re

from . import _topi


@tvm.register_func("tvm_graph.compute.assign")
def compute_assign(a, b):
    b.copyto(a)


@tvm.register_func("tvm_graph.tensor.assign")
def tensor_assign(a, b):
    return tvm.extern(
        a.shape, [a, b],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.assign", ins[0], ins[1]),
        name="assign")


@tvm.register_func("tvm_graph.compute.zeros")
def compute_zero(x):
    tvm.nd.array(np.zeros(x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.zeros")
def tensor_zeros(shape):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    return tvm.extern(
        shape, [],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.zeros", outs[0]),
        name="zeros",
        dtype=[tvm.float32])


@tvm.register_func("tvm_graph.tensor.zeros_like")
def tensor_zeros_like(x):
    return tvm.extern(
        x.shape, [],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.zeros", outs[0]),
        name="zeros_like",
        dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.ones")
def compute_ones(x):
    tvm.nd.array(np.ones(x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.ones")
def tensor_ones(shape):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    return tvm.extern(
        shape, [],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.ones", outs[0]),
        name="ones",
        dtype=[tvm.float32])


@tvm.register_func("tvm_graph.tensor.ones_like")
def tensor_ones_like(x):
    return tvm.extern(
        x.shape, [],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.ones", outs[0]),
        name="ones_like",
        dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.normal")
def compute_normal(x, loc, scale):
    tvm.nd.array(
        np.random.normal(loc, scale, x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.normal")
def tensor_normal(shape, loc, scale):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    loc = float(loc)
    scale = float(scale)
    return tvm.extern(
        shape, [],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.normal", outs[0], loc, scale),
        name="normal",
        dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.equal")
def compute_equal(a, b):
    op = tvm.compute(a.shape,
                     lambda *i: a(*i).equal(b(*i)).astype(tvm.float32))
    return op


@tvm.register_func("tvm_graph.compute.argmax")
def compute_argmax(x, idx, axis):
    np_idx = np.argmax(x.asnumpy(), axis=axis)
    tvm.nd.array(np_idx.astype(np.float32)).copyto(idx)


@tvm.register_func("tvm_graph.tensor.argmax")
def tensor_argmax(x, axis):
    oshape = []
    for i in range(len(x.shape)):
        if i == axis:
            continue
        else:
            oshape.append(x.shape[i])
    return tvm.extern(
        [x.shape[0]], [x],
        lambda ins, outs: tvm.call_packed("tvm_graph.compute.argmax", ins[0], outs[0], axis),
        name="argmax")


@tvm.register_func("tvm_graph.compute.red_sum")
def compute_red_sum(x, axis, keepdims=False):
    tmp = re.findall('\d+', axis)
    axis = None
    if len(tmp) > 0:
        axis = [int(i) for i in tmp]
    return _topi.reduction.sum(x, axis=axis, keepdims=keepdims)


@tvm.register_func("tvm_graph.compute.red_sum_bwd")
def compute_red_sum_bwd(x, x_ori):
    shape = x_ori.shape
    return _topi.broadcast_to(x, shape)


@tvm.register_func("tvm_graph.compute.red_mean")
def compute_red_mean(x, axis, keepdims=False):
    tmp = re.findall('\d+', axis)
    axis = None
    if len(tmp) > 0:
        axis = [int(i) for i in tmp]
    factor = tvm.convert(1)
    if axis != None:
        for i in axis:
            factor *= x.shape[i]
    else:
        for i in x.shape:
            factor *= i
    red_sum = _topi.reduction.sum(x, axis=axis, keepdims=keepdims)
    return tvm.compute(red_sum.shape, lambda *i: red_sum(*i) / factor)


@tvm.register_func("tvm_graph.compute.red_mean_bwd")
def compute_red_mean_bwd(x, x_ori, axis):
    shape = x_ori.shape
    tmp = re.findall('\d+', axis)
    axis = None
    if len(tmp) > 0:
        axis = [int(i) for i in tmp]
    factor = tvm.convert(1)
    if axis != None:
        for i in axis:
            factor *= shape[i]
    else:
        for i in shape:
            factor *= i
    broad_sum = _topi.broadcast_to(x, shape)
    return tvm.compute(broad_sum.shape, lambda *i: broad_sum(*i) / factor)


@tvm.register_func("tvm_graph.compute.softmax")
def compute_softmax(x):
    return _topi.nn.softmax(x)


@tvm.register_func("tvm_graph.compute.softmax_bwd")
def compute_softmax_bwd(out_grad, out_data):
    tmp1 = tvm.compute(out_data.shape, lambda *i: out_grad(*i) * out_data(*i))
    tmp2 = _topi.reduction.sum(tmp1, axis=1, keepdims=True)
    return tvm.compute(
        out_data.shape,
        lambda i, j: out_data[i][j] * (out_grad[i][j] - tmp2[i][0]))


@tvm.register_func("tvm_graph.compute.relu")
def compute_relu(x):
    return _topi.nn.elemwise.relu(x)


@tvm.register_func("tvm_graph.compute.relu_bwd")
def compute_relu_bwd(outgrad, y):
    return tvm.compute(
        y.shape, lambda *i: outgrad(*i) * (y(*i) > 0).astype(tvm.float32))


@tvm.register_func("tvm_graph.compute.global_pool")
def compute_global_pool(x):
    return _topi.nn.global_avg_pool(x)


@tvm.register_func("tvm_graph.compute.global_pool_bwd")
def compute_global_pool_bwd(outgrad, indata):
    batch, channel, height, width = indata.shape
    tmp = _topi.broadcast_to(indata.shape, outgrad)
    return tvm.compute(tmp.shape, lambda *i: tmp(*i) / height / width)