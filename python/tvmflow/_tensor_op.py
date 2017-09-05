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


@tvm.register_func("tvm_graph.compute.assign")
def compute_assign(a, b):
    b.copyto(a)


@tvm.register_func("tvm_graph.tensor.assign")
def tensor_assign(a, b):
    return tvm.extern(a.shape, [a, b], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.assign", ins[0], ins[1]), name="assign")


@tvm.register_func("tvm_graph.compute.zeros")
def compute_zero(x):
    tvm.nd.array(np.zeros(x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.zeros")
def tensor_zeros(shape):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    return tvm.extern(shape, [], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.zeros", outs[0]), name="zeros",  dtype=[tvm.float32])


@tvm.register_func("tvm_graph.tensor.zeros_like")
def tensor_zeros_like(x):
    return tvm.extern(x.shape, [], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.zeros", outs[0]), name="zeros_like",  dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.ones")
def compute_ones(x):
    tvm.nd.array(np.ones(x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.ones")
def tensor_ones(shape):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    return tvm.extern(shape, [], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.ones", outs[0]), name="ones",  dtype=[tvm.float32])


@tvm.register_func("tvm_graph.tensor.ones_like")
def tensor_ones_like(x):
    return tvm.extern(x.shape, [], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.ones", outs[0]), name="ones_like",  dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.normal")
def compute_normal(x, loc, scale):
    tvm.nd.array(np.random.normal(
        loc, scale, x.shape).astype(x.dtype)).copyto(x)


@tvm.register_func("tvm_graph.tensor.normal")
def tensor_normal(shape, loc, scale):
    ss = re.findall('\d+', shape)
    shape = [int(x) for x in ss]
    loc = float(loc)
    scale = float(scale)
    return tvm.extern(shape, [], lambda ins, outs: tvm.call_packed(
        "tvm_graph.compute.normal", outs[0], loc, scale), name="normal",  dtype=[tvm.float32])


@tvm.register_func("tvm_graph.compute.equal")
def compute_equal(a, b):
    op = tvm.compute(
        a.shape, lambda *i: a(*i).equal(b(*i)).astype(tvm.float32))
    return op
