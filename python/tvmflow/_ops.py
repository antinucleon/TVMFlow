"""Wrapping of certain ops for positional arguments.
Mainly because NNVM accepts kwargs for some additional arguments,
while TF sometimes support positional ops.
"""
from __future__ import absolute_import as _abs
from ._nnvm import symbol
from ._nnvm import _symbol_internal

from symbol import *

import tvm


@tvm.register_func("tvm_graph.compute.add")
def compute_add(a, b):
    return tvm.compute(a.shape, lambda *i: a(*i) + b(*i))


@tvm.register_func("tvm_graph.compute.exp")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.exp(a(*i)))


@tvm.register_func("tvm_graph.schedule.extern")
def schedule_extern(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    return s


@tvm.register_func("tvm_graph.schedule.ewise")
def schedule_ewise(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    # for x in outs:
    #    bx, tx = s[x].split(x.op.axis[0], factor=2)
    #    s[x].bind(bx, tvm.thread_axis("blockIdx.x"))
    #    s[x].bind(tx, tvm.thread_axis("threadIdx.x"))
    tvm.schedule.AutoInlineElemWise(s)
    return s
