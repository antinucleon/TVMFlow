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
    return tvm.intrin.call_pure_intrin(x.dtype, "pow", x, y)


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
    op = tvm.compute(
        a.shape,
        lambda *i: a(*i).equal(b(*i)).astype(tvm.float32),
        tag="ewise")
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
    return _topi.reduction.sum(x, axis=axis, keepdims=keepdims, tag="reduce")


@tvm.register_func("tvm_graph.compute.red_sum_bwd")
def compute_red_sum_bwd(x, x_ori):
    shape = x_ori.shape
    return _topi.broadcast_to(x, shape, tag="bcast")


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
    red_sum = _topi.reduction.sum(
        x, axis=axis, keepdims=keepdims, tag="reduce")
    return tvm.compute(
        red_sum.shape, lambda *i: red_sum(*i) / factor, tag="ewise")


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
    broad_sum = _topi.broadcast_to(x, shape, tag="bcast")
    return tvm.compute(
        broad_sum.shape, lambda *i: broad_sum(*i) / factor, tag="ewise")


@tvm.register_func("tvm_graph.compute.softmax")
def compute_softmax(x):
    return _topi.nn.softmax(x)


@tvm.register_func("tvm_graph.compute.softmax_bwd")
def compute_softmax_bwd(out_grad, out_data):
    m, n = out_grad.shape
    tmp1 = tvm.compute(
        out_data.shape, lambda *i: out_grad(*i) * out_data(*i), tag="ewise")
    k = tvm.reduce_axis((0, n), name='k')
    tmp2 = tvm.compute((m), lambda i: tvm.sum(tmp1[i, k], axis=[k]))
    #tmp2 = _topi.reduction.sum(tmp1, axis=1, keepdims=True)
    return tvm.compute(
        out_data.shape,
        lambda i, j: out_data[i][j] * (out_grad[i][j] - tmp2[i]),
        tag="ewise")


@tvm.register_func("tvm_graph.compute.relu")
def compute_relu(x):
    return _topi.nn.elemwise.relu(x)


@tvm.register_func("tvm_graph.compute.relu_bwd")
def compute_relu_bwd(outgrad, y):
    return tvm.compute(
        y.shape,
        lambda *i: outgrad(*i) * (y(*i) > 0).astype(tvm.float32),
        tag="ewsie")


@tvm.register_func("tvm_graph.compute.global_pool")
def compute_global_pool(x):
    return _topi.nn.global_avg_pool(x)


@tvm.register_func("tvm_graph.compute.global_pool_bwd")
def compute_global_pool_bwd(outgrad, indata):
    batch, channel, height, width = indata.shape
    tmp = _topi.broadcast_to(outgrad, indata.shape, tag="bcast")
    return tvm.compute(
        tmp.shape, lambda *i: tmp(*i) / height / width, tag="ewise")


@tvm.register_func("tvm_graph.compute.flatten")
def compute_flatten(x):
    # flatten output from global pool
    ishape = x.shape
    assert len(ishape) == 4
    return tvm.compute(
        (ishape[0], ishape[1]), lambda i, j: x[i, j, 0, 0], tag="ewise")


@tvm.register_func("tvm_graph.compute.flatten_bwd")
def compute_flatten_bwd(x):
    # return from 2d to 4d
    ishape = x.shape
    return tvm.compute(
        (ishape[0], ishape[1], 1, 1), lambda i, j, k, l: x[i, j], tag="ewise")


@tvm.register_func("tvm_graph.compute.bn_train")
def compute_bn_train(x, gamma, beta, eps):
    batch, in_channel, in_height, in_width = x.shape
    factor = batch * in_height * in_channel
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    mean = tvm.compute((1, in_channel, 1, 1),
                       lambda n, c, h, w: tvm.sum(x[rb, c, rh, rw] / factor, axis=[rb, rh, rw]), tag="reduce")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    # change when we have pow
    var = tvm.compute((1, in_channel, 1, 1),
                      lambda n, c, h, w: tvm.sum(((x[rb, c, rh, rw] - mean[0, c, 0, 0]) * (x[rb, c, rh, rw] - mean[0, c, 0, 0]))
                                                 / factor, axis=[rb, rh, rw]), tag="reduce")
    out = tvm.compute(x.shape,
                      lambda n, c, h, w:
                      gamma[0, c, 0, 0] * (x[n, c, h, w] - mean[0, c, 0, 0]) /
                      tvm.sqrt(var[0, c, 0, 0] + eps) + beta[0, c, 0, 0], tag="ewise")
    return out


@tvm.register_func("tvm_graph.compute.bn_bwd_data")
def compute_bn_bwd(outgrad, x, gamma, eps):
    batch, in_channel, in_height, in_width = x.shape
    factor = batch * in_height * in_width
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    mean = tvm.compute((1, in_channel, 1, 1),
                       lambda n, c, h, w: tvm.sum(x[rb, c, rh, rw] / factor, axis=[rb, rh, rw]), tag="reduce")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    # change when we have pow
    var = tvm.compute((1, in_channel, 1, 1),
                      lambda n, c, h, w: tvm.sum(((x[rb, c, rh, rw] - mean[0, c, 0, 0]) * (x[rb, c, rh, rw] - mean[0, c, 0, 0]))
                                                 / factor, axis=[rb, rh, rw]), tag="reduce")    # grad
    g_tmp = tvm.compute(
        outgrad.shape,
        lambda n, c, h, w: outgrad[n, c, h, w] * gamma[0, c, 0, 0],
        tag="ewise")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_var = tvm.compute(gamma.shape,
                        lambda n, c, h, w: tvm.sum(g_tmp[n, c, h, w] *
                                                   (x[n, c, h, w] - mean[0, c, 0, 0]) * (-0.5) *
                                                   1.0 /
                                                   tvm.sqrt((var[0, c, 0, 0] + eps) *
                                                            (var[0, c, 0, 0] + eps) *
                                                            (var[0, c, 0, 0] + eps)), axis=[rb, rh, rw]), tag="reduce")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_mean_1 = tvm.compute(gamma.shape,
                           lambda n, c, h, w: tvm.sum(-1 * g_tmp[n, c, h, w] /
                                                      tvm.sqrt(var[0, c, 0, 0] + eps), axis=[rb, rh, rw]), tag="reduce")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_mean_2 = tvm.compute(gamma.shape,
                           lambda n, c, h, w: tvm.sum(-2 * (x[n, c, h, w] - mean[0, c, 0, 0]) / factor,
                                                      axis=[rb, rh, rw]), tag="reduce")
    g_mean = tvm.compute(
        gamma.shape,
        lambda n, c, h, w: g_mean_1[0, c, 0, 0] + g_var[0, c, 0, 0] * g_mean_2[0, c, 0, 0],
        tag="ewsie")
    g_x = tvm.compute(x.shape,
                      lambda n, c, h, w: -1 * g_tmp[n, c, h, w] /
                      tvm.sqrt(var[0, c, 0, 0] + eps) +
                      g_var[0, c, 0, 0] * 2 * (x[n, c, h, w] - mean[0, c, 0, 0]) /
                      factor + g_mean[0, c, 0, 0] / factor, tag="ewise")

    return g_x


@tvm.register_func("tvm_graph.compute.bn_bwd_gamma")
def compute_bn_bwd_gamma(outgrad, x, gamma, eps):
    batch, in_channel, in_height, in_width = x.shape
    factor = batch * in_height * in_width
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    mean = tvm.compute((1, in_channel, 1, 1),
                       lambda n, c, h, w: tvm.sum(x[rb, c, rh, rw] / factor, axis=[rb, rh, rw]), tag="reduce")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    # change when we have pow
    var = tvm.compute((1, in_channel, 1, 1),
                      lambda n, c, h, w: tvm.sum(((x[rb, c, rh, rw] - mean[0, c, 0, 0]) * (x[rb, c, rh, rw] - mean[0, c, 0, 0]))
                                                 / factor, axis=[rb, rh, rw]), tag="reduce")    # grad
    tmp = tvm.compute(x.shape,
                      lambda n, c, h, w:
                      (x[n, c, h, w] - mean[0, c, 0, 0]) /
                      tvm.sqrt(var[0, c, 0, 0] + eps), tag="ewise")
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_gamma = tvm.compute(gamma.shape,
                          lambda n, c, h, w: tvm.sum(outgrad[n, c, h, w] * gamma[0, c, 0, 0] * tmp[n, c, h, w], axis=[rb, rh, rw]), tag="reduce")
    return g_gamma


@tvm.register_func("tvm_graph.compute.conv2d")
def compute_conv2d_nchw(Input, Filter, stride, padding):
    return _topi.nn.conv2d_nchw(Input, Filter, stride, padding)


@tvm.register_func("tvm_graph.shape_infer.conv2d")
def infer_conv2d_shape(input_shape, weight_shape, stride, padding):
    ss = re.findall('\d+', input_shape)
    batch, in_channel, in_height, in_width = [int(x) for x in ss]
    ss = re.findall('\d+', weight_shape)
    num_filter, channel, kernel_h, kernel_w = [int(x) for x in ss]
    stride_h = stride_w = int(stride)
    pad_top, pad_left, pad_down, pad_right = _topi.nn.get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = _topi.util.simplify(
        (in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = _topi.util.simplify(
        (in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    return str((batch, out_channel, out_height, out_width))


@tvm.register_func("tvm_graph.compute.conv2d_bwd_data")
def compute_conv2d_nchw_bwd_data(outgrad, Input, weight, stride, padding):
    batch, out_c, out_h, out_w = outgrad.shape
    _, in_c, in_h, in_w = Input.shape
    out_ch, in_ch, kernel_h, kernel_w = weight.shape
    stride_h = stride_w = int(stride)
    # fwd pad
    fpad_top, fpad_left, fpad_bottom, fpad_right = _topi.nn.get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # padding params in backward propagation
    bpad_top = kernel_h - 1 - fpad_top
    bpad_bottom = (kernel_h - 1 - fpad_bottom) + (stride_h - 1)
    bpad_left = kernel_w - 1 - fpad_left
    bpad_right = (kernel_w - 1 - fpad_right) + (stride_w - 1)

    padded_out_grad = _topi.nn.pad(
        outgrad, [0, 0, bpad_top, bpad_left], [0, 0, bpad_bottom, bpad_right],
        name='padded_out_grad')

    rc = tvm.reduce_axis((0, out_ch), name="rb")
    rh = tvm.reduce_axis((0, kernel_h), name="rh")
    rw = tvm.reduce_axis((0, kernel_w), name="rw")

    grad = tvm.compute(
        (batch, in_c, in_h, in_w),
        lambda n, c, h, w: tvm.sum(
            padded_out_grad[n, rc, h + rh, w + rw] *
            weight[rc, c, kernel_h - 1 - rh, kernel_w - 1 - rw],
            axis=[rc, rh, rw]
        )
    )

    return grad


@tvm.register_func("tvm_graph.compute.conv2d_bwd_weight")
def compute_conv2d_nchw_bwd_weight(outgrad, Input, weight, stride, padding):
    batch, out_c, out_h, out_w = outgrad.shape
    _, in_c, in_h, in_w = Input.shape
    out_ch, in_ch, kernel_h, kernel_w = weight.shape

    stride_h = stride_w = int(stride)
    # pad
    pad_top, pad_left, pad_down, pad_right = _topi.nn.get_pad_tuple(
        padding, (kernel_h, kernel_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    padded_in = _topi.nn.pad(Input, pad_before, pad_after, name="pad_temp")

    rb = tvm.reduce_axis((0, batch), name="rb")
    rh = tvm.reduce_axis((0, out_h), name="rh")
    rw = tvm.reduce_axis((0, out_w), name="rw")

    gw = tvm.compute(weight.shape,
                     lambda co, ci, fh, fw: tvm.sum(
                         outgrad[rb, co, rh, rw] *
                         padded_in[rb, ci, fh + rh * stride_h,
                                   fw + rw * stride_w], axis=[rb, rh, rw]
                     ), tag="reduce")
    return gw


@tvm.register_func("tvm_graph.compute.bias4d")
def compute_bias4d(data, bias):
    batch, in_channel, in_height, in_width = data.shape
    return tvm.compute(
        data.shape,
        lambda n, c, h, w: data[n, c, h, w] + bias[0, c, 0, 0],
        tag="ewise")


@tvm.register_func("tvm_graph.compute.bias4d_bwd")
def compute_bias4d_bwd(out_grad, bias):
    batch, in_channel, in_height, in_width = out_grad.shape
    factor = batch * in_height * in_width
    rb = tvm.reduce_axis((0, batch))
    rh = tvm.reduce_axis((0, in_height))
    rw = tvm.reduce_axis((0, in_width))
    g_bias = tvm.compute(
        bias.shape, lambda n, c, h, w: tvm.sum(out_grad[n, c, h, w] / factor, axis=[rb, rh, rw]), tag="reduce")
    return g_bias


@tvm.register_func("tvm_graph.compute.bias2d")
def compute_bias2d(data, bias):
    assert len(data.shape) == 2
    assert len(bias.shape) == 2
    batch, channel = data.shape
    return tvm.compute(
        data.shape, lambda n, c: data[n, c] + bias[0, c], tag="ewise")


@tvm.register_func("tvm_graph.compute.bias2d_bwd")
def compute_bias_bwd(outgrad, bias):
    assert len(outgrad.shape) == 2
    assert len(bias.shape) == 2
    batch, channel = outgrad.shape
    rb = tvm.reduce_axis((0, batch))
    return tvm.compute(
        bias.shape,
        lambda n, c: tvm.sum(outgrad[rb, c] / batch, axis=[rb]),
        tag="reduce")


@tvm.register_func("tvm_graph.compute.indentity")
def compute_indentity(data):
    return tvm.compute(data.shape, lambda *i: data(*i), tag="ewise")
