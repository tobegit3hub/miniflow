# Copyright 2017 The Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains all the basic operations."""

import logging
import math
import numpy as np
import os
import sys

import graph

# Enable swig by environment variable
if os.environ.has_key("ENABLE_SWIG_OP"):
  logging.info("Enable swig operations by environment variable")
  sys.path.append("../")
  import swig.op


class Op(object):
  """The basic class for all operation."""

  def __init__(self):
    pass

  def forward(self):
    raise NotImplementedError

  def grad(self):
    raise NotImplementedError

  def __add__(self, other):
    return AddOp(self, other)

  def __radd__(self, other):
    return AddOp(other, self)

  def __sub__(self, other):
    return MinusOp(self, other)

  def __rsub__(self, other):
    return MinusOp(other, self)

  def __mul__(self, other):
    return MultipleOp(self, other)

  def __rmul__(self, other):
    return MultipleOp(other, self)

  def __div__(self, other):
    return DivideOp(self, other)

  def __rdiv__(self, other):
    return DivideOp(other, self)


class PlaceholderOp(Op):
  """The placeholer operation which value is set when Session.run()"""

  def __init__(self, dtype=None, shape=None, name="Placeholder"):
    # TODO: Use dtype and shape
    self.dtype = dtype
    self.shape = shape
    self.name = name

    # The value is None util Session.run() with feed_dict parameter
    self.value = None

    # TODO: Support other graph instance
    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def set_value(self, value):
    self.value = value

  def get_value(self):
    return self.value

  def forward(self):
    return self.value

  def grad(self):
    return 0


class ConstantOp(Op):
  """The constant operation which contains one initialized value."""

  def __init__(self, value, name="Constant"):
    self.value = value
    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  # TODO: Not allow to set the value

  def get_value(self):
    return self.value

  def forward(self):
    return self.value

  def grad(self):
    return 0


class VariableOp(Op):
  """
  The variable operation which contains one variable. The variable may be
  trainable or not-trainable. This is used to define the machine learning
  models.
  """

  def __init__(self, value, is_trainable=True, name="Variable"):
    self.value = value
    self.is_trainable = is_trainable
    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

    if self.is_trainable:
      self.graph.add_to_trainable_variables_collection(self.name, self)

  def get_value(self):
    return self.value

  def set_value(self, value):
    self.value = value

  def forward(self):
    return self.value

  def grad(self):
    return 1


def test_VariableOp():
  x = 10
  variable = VariableOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class SquareOp(Op):
  def __init__(self, input, name="Square"):
    if not isinstance(input, Op):
      self.op = ConstantOp(input)
    else:
      self.op = input

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if os.environ.has_key("ENABLE_SWIG_OP"):
      result = swig.op.square(self.op.forward())
    else:
      result = pow(self.op.forward(), 2)
    return result

  def grad(self):
    return 2 * self.op.forward()


def test_SquareOp():
  x = 10
  variable = SquareOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class CubicOp(Op):
  def __init__(self, input, name="Cubic"):
    if not isinstance(input, Op):
      self.op = ConstantOp(input)
    else:
      self.op = input

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if os.environ.has_key("ENABLE_SWIG_OP"):
      result = swig.op.cubic(self.op.forward())
    else:
      result = math.pow(self.op.forward(), 3)
    return result

  def grad(self):
    if os.environ.has_key("ENABLE_SWIG_OP"):
      result = swig.op.multiple(3, swig.op.square(self.op.forward()))
    else:
      result = 3 * math.pow(self.op.forward(), 2)
    return result


def test_CubicOp():
  x = 10
  variable = CubicOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


def SigmoidOp(x):
  def __init__(self, x, name="Sigmoid"):
    self.name = name
    self.x = x

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return 1.0 / (1 + np.exp(-x))

  def grad(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return forward(x) / (1 - forward(x))


class AddOp(Op):
  """
  The addition operation which has only two inputs. The input can be
  primitive, ConstantOp, PlaceholerOp, VariableOp or other ops.
  """

  def __init__(self, input1, input2, name="Add"):
    if not isinstance(input1, Op):
      self.op1 = ConstantOp(input1)
    else:
      self.op1 = input1

    if not isinstance(input2, Op):
      self.op2 = ConstantOp(input2)
    else:
      self.op2 = input2

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = self.op1.forward() + self.op2.forward()
    return result

  def grad(self):
    result = self.op1.grad() + self.op2.grad()
    return result


class MinusOp(Op):
  """
  The minus operation.
  """

  def __init__(self, input1, input2, name="Minus"):
    if not isinstance(input1, Op):
      self.op1 = ConstantOp(input1)
    else:
      self.op1 = input1

    if not isinstance(input2, Op):
      self.op2 = ConstantOp(input2)
    else:
      self.op2 = input2

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = self.op1.forward() - self.op2.forward()
    return result

  def grad(self):
    result = self.op1.grad() - self.op2.grad()
    return result


class AddNOp(Op):
  def __init__(self, *inputs):
    # TODO: Support user defined name in the parameter
    self.name = "AddN"

    self.ops = []
    for input in inputs:
      if not isinstance(input, Op):
        input = ConstantOp(input)
      self.ops.append(input)

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = 0
    for op in self.ops:
      result += op.forward()
    return result

  def grad(self):
    result = 0
    for op in self.ops:
      result += op.grad()
    return result


# TODO: Can not support operations like "x * x", only "x * 3"
class MultipleOp(Op):
  def __init__(self, input1, input2, name="Multiple"):
    if not isinstance(input1, Op):
      self.op1 = ConstantOp(input1)
    else:
      self.op1 = input1

    if not isinstance(input2, Op):
      self.op2 = ConstantOp(input2)
    else:
      self.op2 = input2

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = self.op1.forward() * self.op2.forward()
    return result

  def grad(self):
    result = self.op1.grad() * self.op2.grad()
    return result


class MultipleNOp(Op):
  def __init__(self, *inputs):
    self.name = "MultipleN"

    self.ops = []
    for input in inputs:
      if not isinstance(input, Op):
        input = ConstantOp(input)
      self.ops.append(input)

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = 1
    for op in self.ops:
      result *= op.forward()
    return result

  def grad(self):
    result = 1
    for op in self.ops:
      result *= op.grad()
    return result


class DivideOp(Op):
  def __init__(self, input1, input2, name="Divide"):
    if not isinstance(input1, Op):
      self.op1 = ConstantOp(input1)
    else:
      self.op1 = input1

    if not isinstance(input2, Op):
      self.op2 = ConstantOp(input2)
    else:
      self.op2 = input2

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = self.op1.forward() / self.op2.forward()
    return result

  def grad(self):
    result = self.op1.grad() / self.op2.grad()
    return result


class UpdateVariableOp(Op):
  def __init__(self, variableOp, value, name="UpdateVariableOp"):
    self.variableOp = variableOp
    self.value = value
    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    self.variableOp.set_value(self.value)
    return self.value

  # TODO: Add grad() if needed


class UpdateVariableNOp(Op):
  def __init__(self, variableop_value_map, name="UpdateVariableNOp"):
    self.variableop_value_map = variableop_value_map
    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    # TODO: Need to test the loop
    for variableOp, value in enumerate(self.variableop_value_map):
      variableOp.set_value(value)
    return self.variableop_value_map
