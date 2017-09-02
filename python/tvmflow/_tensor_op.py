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


@tvm.register_func("tvm_graph.compute.zeros")
def compute_zero(x):
    tvm.nd.array(np.zeros(x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.zeros")
def tensor_zeros(shape):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    return tvm.extern(shape, [], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.zeros", outs[0]), name="zeros",  dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.assign")
def compute_assign(a, b):
    b.copyto(a)


@tvm.register_func("tvm_graph.tensor.assign")
def tensor_assign(a, b):
    return tvm.extern(a.shape, [a, b], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.assign", ins[0], ins[1]), name="assign")
