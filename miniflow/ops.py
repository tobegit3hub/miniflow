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
    return self.__add__(other)

  def __sub__(self, other):
    return MinusOp(self, other)

  def __rsub__(self, other):
    return MinusOp(other, self)

  def __mul__(self, other):
    return MultipleOp(self, other)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __div__(self, other):
    return DivideOp(self, other)

  def __rdiv__(self, other):
    return DivideOp(other, self)

  def __pow__(self, power, modulo=None):
    return PowerOp(self, power)


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

  def grad(self, partial_derivative_opname=None):
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

  def grad(self, partial_derivative_opname=None):
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

  def grad(self, partial_derivative_opname=None):
    if partial_derivative_opname is None:
      grad = 1
    else:
      if self.name == partial_derivative_opname:
        # Specify to compute this derivative
        grad = 1
      else:
        # Specify to compute other derivative
        grad = 0
    return grad


def test_VariableOp():
  x = 10
  variable = VariableOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class PowerOp(Op):
  def __init__(self, input, power, name="Power"):
    if not isinstance(input, Op):
      self.op = ConstantOp(input)
    else:
      self.op = input

    self.power = power
    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    result = pow(self.op.forward(), self.power)
    return result

  def grad(self, partial_derivative_opname=None):
    if isinstance(self.op, PlaceholderOp) or isinstance(self.op, ConstantOp):
      # op is the constant
      grad = 0
    elif isinstance(self.op, VariableOp):
      # op is the variable
      grad = self.power * pow(self.op.forward(), self.power - 1)
    else:
      # op is other complex operation and use chain rule
      grad = self.power * pow(self.op.forward(), self.power - 1
                              ) * self.op.grad(partial_derivative_opname)

    return grad


class SquareOp(PowerOp):
  def __init__(self, input, name="Square"):
    super(SquareOp, self).__init__(input, 2, name)


class SquareOpOld(Op):
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

  def grad(self, partial_derivative_opname=None):
    if isinstance(self.op, PlaceholderOp) or isinstance(self.op, ConstantOp):
      # op is the constant
      grad = 0
    elif isinstance(self.op, VariableOp):
      # op is the variable
      if os.environ.has_key("ENABLE_SWIG_OP"):
        grad = swig.op.multiple(2, self.op.forward())
      else:
        grad = 2 * self.op.forward()
    else:
      # op is other complex operation and use chain rule
      grad = 2 * self.op.forward() * self.op.grad(partial_derivative_opname)

    return grad


class CubicOp(PowerOp):
  def __init__(self, input, name="Cubic"):
    super(CubicOp, self).__init__(input, 3, name)


class CubicOpOld(Op):
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

  def grad(self, partial_derivative_opname=None):
    if isinstance(self.op, PlaceholderOp) or isinstance(self.op, ConstantOp):
      # op is the constant
      grad = 0
    elif isinstance(self.op, VariableOp):
      # op is the variable
      if os.environ.has_key("ENABLE_SWIG_OP"):
        grad = swig.op.multiple(3, swig.op.square(self.op.forward()))
      else:
        grad = 3 * math.pow(self.op.forward(), 2)
    else:
      # op is other complex operation
      grad = 3 * math.pow(self.op.forward(),
                          2) * self.op.grad(partial_derivative_opname)

    return grad


def test_CubicOp():
  x = 10
  variable = CubicOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


def SigmoidOp(Op):
  def __init__(self, value, name="Sigmoid"):
    pass


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

  def grad(self, partial_derivative_opname=None):
    result = self.op1.grad(partial_derivative_opname) + self.op2.grad(
        partial_derivative_opname)
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

  def grad(self, partial_derivative_opname=None):
    result = self.op1.grad(partial_derivative_opname) - self.op2.grad(
        partial_derivative_opname)
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

  def grad(self, partial_derivative_opname=None):
    result = 0
    for op in self.ops:
      result += op.grad(partial_derivative_opname)
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

  def grad(self, partial_derivative_opname=None):

    if isinstance(self.op1, PlaceholderOp) or isinstance(self.op1, ConstantOp):
      # op1 is the coefficient of this formula
      op1_grad = self.op1.forward()

      if isinstance(self.op2, PlaceholderOp) or isinstance(
          self.op2, ConstantOp):
        # two elements are both constant values
        op2_grad = 0
      else:
        # op2 may has VariableOp
        op2_grad = self.op2.grad(partial_derivative_opname)

    elif isinstance(self.op2, PlaceholderOp) or isinstance(
        self.op2, ConstantOp):
      # op2 is the coefficient of this formula
      op2_grad = self.op2.forward()

      # op1 may has VariableOp
      op1_grad = self.op1.grad(partial_derivative_opname)

    else:
      # op1 and op2 may has VariableOp
      logging.error(
          "Not support complex formula which has multiple VariableOp")
      raise NotImplementedError

    result = op1_grad * op2_grad
    return result


class MultipleNOp(Op):
  """The multiple operation for n inputs."""

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

  def grad(self, partial_derivative_opname=None):
    # TODO: Check the type of op to compute gradients
    result = 1
    for op in self.ops:
      result *= op.grad(partial_derivative_opname)
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

  def grad(self, partial_derivative_opname=None):
    result = self.op1.grad(partial_derivative_opname) / self.op2.grad(
        partial_derivative_opname)
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


def test_SquareOp():
  w = VariableOp(10)
  b = VariableOp(20)
  x = PlaceholderOp(float)
  x.set_value(2.0)
  y = PlaceholderOp(float)
  y.set_value(3.0)

  loss = SquareOp(y - (w * x + b))
  print("w: {}, forward: {}, grad: {}".format(w.get_value(),
                                              loss.forward(),
                                              loss.grad(w.name)))  # 148.0
  print("b: {}, forward: {}, grad: {}".format(b.get_value(),
                                              loss.forward(),
                                              loss.grad(b.name)))  # 74.0


def global_variables_initializer():
  pass

def local_variables_initializer():
  pass
