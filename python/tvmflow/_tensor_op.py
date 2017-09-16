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


def _tvm_pow(x, y):
    return tvm.intrin.call_pure_intrin(x.dtype, "pow", x)


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


@tvm.register_func("tvm_graph.compute.flatten")
def compute_flatten(x):
    # flatten output from global pool
    ishape = x.shape
    assert len(ishape) == 4
    return tvm.compute((ishape[0], ishape[1]), lambda i, j: x[i, j, 0, 0])


@tvm.register_func("tvm_graph.compute.flatten_bwd")
def compute_flatten_bwd(x):
    # return from 2d to 4d
    ishape = x.shape
    return tvm.compute((ishape[0], ishape[1], 1, 1),
                       lambda i, j, k, l: x[i, j])


@tvm.register_func("tvm_graph.compute.bn_train")
def compute_bn_train(x, gamma, beta, eps):
    batch, in_channel, in_height, in_width = x.shape
    factor = batch * in_height * in_channel
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    mean = tvm.compute((1, in_channel, 1, 1),
        lambda n, c, h, w: tvm.sum(x[rb, c, rh, rw] / factor, axis=[rb, rh, rw]))
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    # change when we have pow
    var = tvm.compute((1, in_channel, 1, 1),
        lambda n, c, h, w: tvm.sum(_tvm_pow(x[rb, c, rh, rw] - mean[0, c, 0, 0], 2)
        / factor, axis=[rb, rh, rw]))
    out = tvm.compute(x.shape,
        lambda n, c, h, w:
            gamma[0, c, 0, 0] * (x[n, c, h, w] - mean[0, c, 0, 0]) /
                tvm.sqrt(var[0, c, 0, 0] + eps) + beta[0, c, 0, 0])
    return out


@tvm.register_func("tvm_graph.compute.bn_bwd")
def compute_bn_bwd(outgrad, x, gamma, eps):
    batch, in_channel, in_height, in_width = x.shape
    factor = batch * in_height * in_width
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    mean=tvm.compute((1, in_channel, 1, 1),
        lambda n, c, h, w: tvm.sum(x[rb, i, rh, rw] / factor, axis=[rb, rh, rw]))
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    # change when we have pow
    var = tvm.compute((1, in_channel, 1, 1),
        lambda n, c, h, w: tvm.sum(_tvm_pow(x[rb, c, rh, rw] - mean[0, c, 0, 0], 2)
        / factor, axis=[rb, rh, rw]))    # grad
    g_tmp = tvm.compute(
        outgrad.shape,
        lambda n, c, h, w: outgrad[n, c, h, w] * gamma[0, c, 0, 0])
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_var=tvm.compute(gamma.shape,
        lambda n, c, h, w: tvm.sum(g_tmp[n, c, h, w] *
                                   (x[n, c, h, w] - mean[0, c, 0, 0]) * (-0.5) *
                                   _tvm_pow(var[0, c, 0, 0] + eps, -1.5), axis=[rb, rh, rw]))
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_mean_1=tvm.compute(gamma.shape,
        lambda n, c, h, w: tvm.sum(-1 * g_tmp[n, c, h, w] /
                                   tvm.sqrt(var[0, c, 0, 0] + eps), axis=[rb, rh, rw]))
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_mean_2=tvm.compute(gamma.shape,
        lambda n, c, h, w: tvm.sum(-2 * (x[n, c, h, w] - mean[0, c, 0, 0]) / factor,
         axis=[rb, rh, rw]))
    g_mean = tvm.compute(
        gamma.shape,
        lambda n, c, h, w: g_mean_1[0, c, 0, 0] + g_var[0, c, 0, 0] * g_mean_2[0, c, 0, 0]
    )
    g_x=tvm.compute(x.shape,
        lambda n, c, h, w: -1 * g_tmp[n, c, h, w] /
            tvm.sqrt(var[0, c, 0, 0] + eps) +
            g_var[0, c, 0, 0] * 2 * (x[n, c, h, w] - mean[0, c, 0, 0]) /
             factor + g_mean[0, c, 0, 0] / factor)
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_gamma=tvm.compute(gamma.shape, lambda n, c, h, w: tvm.sum(
        outgrad[n, c, h, w] * g_tmp[n, c, h, w], axis=[rb, rh, rw]))
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_beta = tvm.compute(
        beta.shape, lambda n, c, h, w: tvm.sum(outgrad[n, c, h, w] / factor))
    return g_x, g_gamma, g_beta


"""
@tvm.register_func("tvm_graph.compute.conv2d")
def compute_conv2d_nchw(Input, Filter, stride, padding):
    return _topi.nn.conv2d_nchw(Input, Filter, stride, padding)

@tvm.register_func("tvm_graph.compute.conv2d_bwd_data")
def compute_conv2d_nchw_bwd_data(outgrad, Input, stride, padding):
    padded_out_grad=pad(outgrad, \
                                  [0, bpad_top, bpad_left, 0], \
                                  [0, bpad_bottom, bpad_right, 0], \
                                  name='padded_out_grad')



    rc=tvm.reduce_axis((0, out_ch), name="rb")
    rh=tvm.reduce_axis((0, filter_h), name="rh")
    rw=tvm.reduce_axis((0, filter_w), name="rw")

    grad=tvm.compute(
        (batch, in_c, in_h, in_w),
        lambda n, c, h, w: tvm.sum(
            pad_out_grad[n, c * out_ch + rc, h + rh, w + rw] *
            weight[rc, c, filter_h - 1 - rh, filter_w - 1 - rw],
            axis=[rc, rh, rw]
        )
    )

    return grad

@tvm.register_func("tvm_graph.compute.conv2d_bwd_weight")
def compute_conv2d_nchw_bwd_data(outgrad, Input, weight_shape, stride, padding):
    batch, out_c, out_h, out_w=outgrad.shape
    out_ch, in_ch, fh, fw=weight_shape
    stride_h, stride_w=stride
    # pad
    rb=tvm.reduce_axis((0, batch), name="rb")
    rh=tvm.reduce_axis((0, out_h), name="rh")
    rw=tvm.reduce_axis((0, out_w), name="rw")

    gw=tvm.compute(weight_shape,
        lambda m, c, fh, fw: tvm.sum(
            outgrad[rb, c * out_ch + m % out_ch, rh, rw] *
            padded_in[rb, c, fh + rh * stride_h, fw + rw * stride_w], axis=[rb, rh, rw])
        )
    return gw
"""