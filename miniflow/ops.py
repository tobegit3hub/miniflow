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

from abc import ABCMeta, abstractmethod
import logging
import math
import os
import sys

#from . import graph
import graph

# Enable swig by environment variable
if "ENABLE_SWIG_OP" in os.environ:
  logging.info("Enable swig operations by environment variable")
  sys.path.append("../")
  import swig.op


class Op(object):
  """The basic class for all operation."""

  def __init__(self, name="Op"):
    # Be compatiable with TensorFlow to remove underline
    self.name = name

  def get_name(self):
    return self.name

  def set_name(self, name):
    self.name = name

  @abstractmethod
  def forward(self):
    # TODO: No need to implement in abstract method
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
    super(PlaceholderOp, self).__init__(name)
    # TODO: Use dtype and shape
    self._dtype = dtype
    self._shape = shape

    # The value is None util Session.run() with feed_dict parameter
    self._value = None

    # TODO: Support other graph instance
    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def set_value(self, value):
    self._value = value

  def get_value(self):
    return self._value

  def forward(self):
    return self._value

  def grad(self, partial_derivative_opname=None):
    return 0


class ConstantOp(Op):
  """The constant operation which contains one initialized value."""

  def __init__(self, value, name="Constant"):
    super(ConstantOp, self).__init__(name)
    self._value = value

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  # TODO: Not allow to set the value

  def get_value(self):
    return self._value

  def forward(self):
    return self._value

  def grad(self, partial_derivative_opname=None):
    return 0


class VariableOp(Op):
  """
  The variable operation which contains one variable. The variable may be
  trainable or not-trainable. This is used to define the machine learning
  models.
  """

  def __init__(self, value, is_trainable=True, name="Variable"):
    super(VariableOp, self).__init__(name)
    self._value = value
    self._is_trainable = is_trainable

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

    if self._is_trainable:
      self._graph.add_to_trainable_variables_collection(self.get_name(), self)

  def get_value(self):
    return self._value

  def set_value(self, value):
    self._value = value

  def forward(self):
    return self._value

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


class PowerOp(Op):
  def __init__(self, input, power, name="Power"):
    super(PowerOp, self).__init__(name)

    if not isinstance(input, Op):
      self._op = ConstantOp(input)
    else:
      self._op = input

    self._power = power

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    result = pow(self._op.forward(), self._power)
    return result

  def grad(self, partial_derivative_opname=None):
    if isinstance(self._op, PlaceholderOp) or isinstance(self._op, ConstantOp):
      # op is the constant
      grad = 0
    elif isinstance(self._op, VariableOp):
      # op is the variable
      grad = self._power * pow(self._op.forward(), self._power - 1)
    else:
      # op is other complex operation and use chain rule
      grad = self._power * pow(self._op.forward(), self._power - 1
                               ) * self._op.grad(partial_derivative_opname)

    return grad


class SquareOp(PowerOp):
  def __init__(self, input, name="Square"):
    super(SquareOp, self).__init__(input, 2, name)


class SquareOpOld(Op):
  # TODO: Deprecated op

  def __init__(self, input, name="Square"):
    if not isinstance(input, Op):
      self.op = ConstantOp(input)
    else:
      self.op = input

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if "ENABLE_SWIG_OP" in os.environ:
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
      if "ENABLE_SWIG_OP" in os.environ:
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
  # TODO: Deprecated op

  def __init__(self, input, name="Cubic"):
    if not isinstance(input, Op):
      self.op = ConstantOp(input)
    else:
      self.op = input

    self.name = name

    self.graph = graph.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if "ENABLE_SWIG_OP" in os.environ:
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
      if "ENABLE_SWIG_OP" in os.environ:
        grad = swig.op.multiple(3, swig.op.square(self.op.forward()))
      else:
        grad = 3 * math.pow(self.op.forward(), 2)
    else:
      # op is other complex operation
      grad = 3 * math.pow(self.op.forward(),
                          2) * self.op.grad(partial_derivative_opname)

    return grad


def SigmoidOp(Op):
  # TODO: Need to implement the forward and grad functions

  def __init__(self, input, name="Sigmoid"):
    super(SigmoidOp, self).__init__(name)


class AddOp(Op):
  """
  The addition operation which has only two inputs. The input can be
  primitive, ConstantOp, PlaceholerOp, VariableOp or other ops.
  """

  def __init__(self, input1, input2, name="Add"):
    super(AddOp, self).__init__(name)

    if not isinstance(input1, Op):
      self._op1 = ConstantOp(input1)
    else:
      self._op1 = input1

    if not isinstance(input2, Op):
      self._op2 = ConstantOp(input2)
    else:
      self._op2 = input2

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    result = self._op1.forward() + self._op2.forward()
    return result

  def grad(self, partial_derivative_opname=None):
    result = self._op1.grad(partial_derivative_opname) + self._op2.grad(
        partial_derivative_opname)
    return result


class MinusOp(Op):
  """
  The minus operation.
  """

  def __init__(self, input1, input2, name="Minus"):
    super(MinusOp, self).__init__(name)

    if not isinstance(input1, Op):
      self._op1 = ConstantOp(input1)
    else:
      self._op1 = input1

    if not isinstance(input2, Op):
      self._op2 = ConstantOp(input2)
    else:
      self._op2 = input2

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    result = self._op1.forward() - self._op2.forward()
    return result

  def grad(self, partial_derivative_opname=None):
    result = self._op1.grad(partial_derivative_opname) - self._op2.grad(
        partial_derivative_opname)
    return result


class AddNOp(Op):
  def __init__(self, *inputs):
    # TODO: Deprecated op
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
    super(MultipleOp, self).__init__(name)

    if not isinstance(input1, Op):
      self._op1 = ConstantOp(input1)
    else:
      self._op1 = input1

    if not isinstance(input2, Op):
      self._op2 = ConstantOp(input2)
    else:
      self._op2 = input2

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    result = self._op1.forward() * self._op2.forward()
    return result

  def grad(self, partial_derivative_opname=None):

    if isinstance(self._op1, PlaceholderOp) or isinstance(
        self._op1, ConstantOp):
      # op1 is the coefficient of this formula
      op1_grad = self._op1.forward()

      if isinstance(self._op2, PlaceholderOp) or isinstance(
          self._op2, ConstantOp):
        # two elements are both constant values
        op2_grad = 0
      else:
        # op2 may has VariableOp
        op2_grad = self._op2.grad(partial_derivative_opname)

    elif isinstance(self._op2, PlaceholderOp) or isinstance(
        self._op2, ConstantOp):
      # op2 is the coefficient of this formula
      op2_grad = self._op2.forward()

      # op1 may has VariableOp
      op1_grad = self._op1.grad(partial_derivative_opname)

    else:
      # op1 and op2 may has VariableOp
      logging.error(
          "Not support complex formula which has multiple VariableOp")
      raise NotImplementedError

    result = op1_grad * op2_grad
    return result


class MultipleNOp(Op):
  # TODO: Deprecated op
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
    super(DivideOp, self).__init__(name)

    if not isinstance(input1, Op):
      self._op1 = ConstantOp(input1)
    else:
      self._op1 = input1

    if not isinstance(input2, Op):
      self._op2 = ConstantOp(input2)
    else:
      self._op2 = input2

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    result = self._op1.forward() / self._op2.forward()
    return result

  def grad(self, partial_derivative_opname=None):
    result = self._op1.grad(partial_derivative_opname) / self._op2.grad(
        partial_derivative_opname)
    return result


class UpdateVariableOp(Op):
  # TODO: Deprecated op

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
  # TODO: Deprecated op

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


class GlobalVariablesInitializerOp(Op):
  def __init__(self, name="GlobalVariablesInitializer"):
    super(GlobalVariablesInitializerOp, self).__init__(name)

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    pass

  def grad(self):
    raise NotImplementedError


class LocalVariablesInitializerOp(Op):
  def __init__(self, name="LocalVariablesInitializer"):
    super(LocalVariablesInitializerOp, self).__init__(name)

    self._graph = graph.get_default_graph()
    self._graph.add_to_graph(self)

  def forward(self):
    pass

  def grad(self):
    raise NotImplementedError


def get_variable(name="Variable",
                 value=None,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 reuse=None,
                 trainable=True):
  # TODO: Support default graph only
  _graph = graph.get_default_graph()

  if name in _graph.get_name_op_map():
    return _graph.get_name_op_map()[name]
  else:
    variable = VariableOp(value=value, name=name)
    return variable
