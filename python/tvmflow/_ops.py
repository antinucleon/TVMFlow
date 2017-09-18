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
        return tvm.compute(a.shape, lambda *i: a(*i) + b(*i), tag="ewise")
    elif ufunc == 1:
        return tvm.compute(a.shape, lambda *i: a(*i) - b(*i), tag="ewise")
    elif ufunc == 2:
        if len(a.shape) == 1 and len(b.shape) == 2:
            return tvm.compute(
                b.shape, lambda i, j: a[0] * b[i, j], tag="ewise")
        else:
            return tvm.compute(a.shape, lambda *i: a(*i) * b(*i), tag="ewise")
    elif ufunc == 3:
        return tvm.compute(a.shape, lambda *i: a(*i) / b(*i), tag="ewise")
    else:
        raise Exception("Unknown ufunc")


@tvm.register_func("tvm_graph.compute.ufunc_scalar")
def compute_ufunc(a, b, ufunc):
    if ufunc == 0:
        return tvm.compute(a.shape, lambda *i: a(*i) + b, tag="ewise")
    elif ufunc == 1:
        return tvm.compute(a.shape, lambda *i: a(*i) - b, tag="ewise")
    elif ufunc == 2:
        return tvm.compute(a.shape, lambda *i: a(*i) * b, tag="ewise")
    elif ufunc == 3:
        return tvm.compute(a.shape, lambda *i: a(*i) / b, tag="ewise")
    else:
        raise Exception("Unknown ufunc")


@tvm.register_func("tvm_graph.compute.rsub")
def compute_ufunc(a, b):
    return tvm.compute(a.shape, lambda *i: b - a(*i), tag="ewise")


@tvm.register_func("tvm_graph.compute.exp")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.exp(a(*i)), tag="ewise")


@tvm.register_func("tvm_graph.compute.log")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.log(a(*i)), tag="ewise")


@tvm.register_func("tvm_graph.compute.sqrt")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.sqrt(a(*i)), tag="ewise")


@tvm.register_func("tvm_graph.compute.pow")
def compute_exp(a, b):
    return tvm.compute(a.shape, lambda *i: tvm.pow(a(*i), b(*i)), tag="ewise")


@tvm.register_func("tvm_graph.compute.rpow_scalar")
def compute_exp(a, b):
    return tvm.compute(a.shape, lambda *i: tvm.pow(b, a(*i)), tag="ewise")


@tvm.register_func("tvm_graph.compute.mat_trans")
def compute_mat_trans(x):
    assert len(x.shape) == 2
    return tvm.compute(
        [x.shape[1], x.shape[0]], lambda i, j: x[j][i], tag="ewise")


@tvm.register_func("tvm_graph.compute.matmul")
def compute_matmul(data, weight):
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim fully_connected"
    batch, in_dim = data.shape
    _, out_dim = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    return tvm.compute(
        (batch, out_dim),
        lambda i, j: tvm.sum(data[i][k] * weight[k][j], axis=k),
        tag="matmul")


""


@tvm.register_func("tvm_graph.schedule.extern")
def schedule_extern(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.ewise")
def schedule_ewise(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineElemWise(s)
    if target == "metal":
        for C in outs:
            b1, b2, b3 = 8, 64, 8
            vectorize = 4
            block_x = tvm.thread_axis("blockIdx.x")
            block_y = tvm.thread_axis("blockIdx.y")
            thread_x = tvm.thread_axis("threadIdx.x")
            thread_y = tvm.thread_axis("threadIdx.y")
            # fuse all the inner indices so we can just do a 2d tile
            fused_axis = s[C].fuse(*C.op.axis)

            oi1, ii = s[C].split(fused_axis, factor=b1 * b2 * b3 * vectorize)
            oi2, ii = s[C].split(ii, factor=b2 * b3 * vectorize)
            oi3, ii = s[C].split(ii, factor=b3 * vectorize)
            oi4, ii = s[C].split(ii, factor=vectorize)

            s[C].bind(oi1, block_x)
            s[C].bind(oi2, block_y)
            s[C].bind(oi3, thread_x)
            s[C].bind(oi4, thread_y)
            s[C].vectorize(ii)
    return s


@tvm.register_func("tvm_graph.schedule.matmul")
def schedule_matmul(outs, target):
    s = tvm.create_schedule([x.op for x in outs])

    def schedule(A, B, C, k):
        AA = s.cache_read(A, "shared", [C])
        BB = s.cache_read(B, "shared", [C])
        AL = s.cache_read(AA, "local", [C])
        BL = s.cache_read(BB, "local", [C])
        CC = s.cache_write(C, "local")

        scale = 4
        num_thread = 32
        block_factor = scale * num_thread
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
        thread_yz = tvm.thread_axis((0, 2), "vthread", name="vy")

        by, yi = s[C].split(C.op.axis[0], factor=block_factor)
        bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
        s[C].bind(by, block_y)
        s[C].bind(bx, block_x)
        s[C].reorder(by, bx, yi, xi)

        tyz, yi = s[C].split(yi, nparts=2)
        ty, yi = s[C].split(yi, nparts=num_thread)
        txz, xi = s[C].split(xi, nparts=2)
        tx, xi = s[C].split(xi, nparts=num_thread)
        s[C].bind(tyz, thread_yz)
        s[C].bind(txz, thread_xz)
        s[C].bind(ty, thread_y)
        s[C].bind(tx, thread_x)
        s[C].reorder(tyz, txz, ty, tx, yi, xi)
        s[CC].compute_at(s[C], tx)

        yo, xo = CC.op.axis
        ko, ki = s[CC].split(k, factor=8)
        kt, ki = s[CC].split(ki, factor=1)
        s[CC].reorder(ko, kt, ki, yo, xo)
        s[AA].compute_at(s[CC], ko)
        s[BB].compute_at(s[CC], ko)
        s[AL].compute_at(s[CC], kt)
        s[BL].compute_at(s[CC], kt)
        # Schedule for A's shared memory load
        ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
        _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread * 4)
        tx, xi = s[AA].split(xi, nparts=num_thread)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        s[AA].vectorize(xi)
        # Schedule for B' shared memory load
        ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
        _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread * 4)
        tx, xi = s[BB].split(xi, nparts=num_thread)
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].vectorize(xi)

    def traverse(OP):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if 'ewise' in OP.tag or 'bcast' in OP.tag:
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule conv2d
        elif 'matmul' in OP.tag:
            A = OP.input_tensors[0]
            B = OP.input_tensors[1]
            C = OP.output(0)
            k = OP.reduce_axis[0]
            schedule(A, B, C, k)

    if target != "llvm":
        if len(outs) == 1:
            traverse(outs[0].op)
        else:
            # bwd
            lhs = outs[0].op
            rhs = outs[1].op
            s[lhs.input_tensors[1]].compute_inline()
            s[rhs.input_tensors[0]].compute_inline()
            traverse(lhs)
            traverse(rhs)
    return s


@tvm.register_func("tvm_graph.schedule.reduction")
def schedule_reduction(outs, target):
    if target == "metal":
        return _topi.cuda.reduction.schedule_reduce(outs)
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.softmax")
def schedule_softmax(outs, target):
    if target == "metal":
        return _topi.cuda.softmax.schedule_softmax(outs)
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.broadcast")
def schedule_broadcast(outs, target):
    if target == "metal":
        return _topi.cuda.broadcast.schedule_broadcast_to(outs)
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.conv")
def schedule_conv(outs, target):
    if target == "metal":
        return _topi.cuda.conv2d_nchw.schedule_conv2d_nchw(outs)
    s = tvm.create_schedule([x.op for x in outs])
    return s