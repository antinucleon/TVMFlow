"""Wrapping of certain ops for positional arguments.
Mainly because NNVM accepts kwargs for some additional arguments,
while TF sometimes support positional ops.
"""
from __future__ import absolute_import as _abs
from ._nnvm import symbol
from ._nnvm import _symbol_internal

from symbol import *

import tvm

from . import _topi


def argmax(x, axis):
    return _symbol_internal._argmax(x, reduction_indices=[axis])


@tvm.register_func("tvm_graph.compute.ufunc")
def compute_ufunc(a, b, ufunc):
    if ufunc == 0:
        return tvm.compute(a.shape, lambda *i: a(*i) + b(*i))
    elif ufunc == 1:
        return tvm.compute(a.shape, lambda *i: a(*i) - b(*i))
    elif ufunc == 2:
        return tvm.compute(a.shape, lambda *i: a(*i) * b(*i))
    elif ufunc == 3:
        return tvm.compute(a.shape, lambda *i: a(*i) / b(*i))
    else:
        raise Exception("Unknown ufunc")


@tvm.register_func("tvm_graph.compute.ufunc_scalar")
def compute_ufunc(a, b, ufunc):
    if ufunc == 0:
        return tvm.compute(a.shape, lambda *i: a(*i) + b)
    elif ufunc == 1:
        return tvm.compute(a.shape, lambda *i: a(*i) - b)
    elif ufunc == 2:
        return tvm.compute(a.shape, lambda *i: a(*i) * b)
    elif ufunc == 3:
        return tvm.compute(a.shape, lambda *i: a(*i) / b)
    else:
        raise Exception("Unknown ufunc")


@tvm.register_func("tvm_graph.compute.rsub")
def compute_ufunc(a, b):
    return tvm.compute(a.shape, lambda *i: b - a(*i))


@tvm.register_func("tvm_graph.compute.exp")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.exp(a(*i)))


@tvm.register_func("tvm_graph.compute.log")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.log(a(*i)))


@tvm.register_func("tvm_graph.compute.sqrt")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.sqrt(a(*i)))


@tvm.register_func("tvm_graph.compute.pow")
def compute_exp(a, b):
    return tvm.compute(a.shape, lambda *i: tvm.pow(a(*i), b(*i)))


@tvm.register_func("tvm_graph.compute.rpow_scalar")
def compute_exp(a, b):
    return tvm.compute(a.shape, lambda *i: tvm.pow(b, a(*i)))


@tvm.register_func("tvm_graph.compute.mat_trans")
def compute_mat_trans(x):
    assert len(x.shape) == 2
    return tvm.compute([x.shape[1], x.shape[0]], lambda i, j: x[j][i])


@tvm.register_func("tvm_graph.compute.matmul")
def compute_matmul(data, weight):
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim fully_connected"
    batch, in_dim = data.shape
    _, out_dim = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    return tvm.compute((batch, out_dim), lambda i, j: \
        tvm.sum(data[i][k] * weight[k][j], axis=k))


@tvm.register_func("tvm_graph.schedule.extern")
def schedule_extern(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.ewise")
def schedule_ewise(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    if target != "llvm":
        for x in outs:
            bx, tx = s[x].split(x.op.axis[0], factor=2)
            s[x].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[x].bind(tx, tvm.thread_axis("threadIdx.x"))
    tvm.schedule.AutoInlineElemWise(s)
    return s


@tvm.register_func("tvm_graph.schedule.matmul")
def schedule_matmul(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.reduction")
def schedule_reduction(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.broadcast")
def schedule_broadcast(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    return s
