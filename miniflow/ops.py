import math
import numpy as np
import os
import sys
sys.path.append("../")

import client
if os.environ.has_key("ENABLE_SWIG_OP"):
  import swig.op


class Op(object):
  def __init__(self):
    pass

  def forward(self):
    pass

  def grad(self):
    pass


class PlaceholderOp(Op):
  def __init__(self, dtype=None, name=None):
    if name == None:
      self.name = "Placeholder"
    else:
      self.name = name

    self.dtype = dtype
    self.value = None

    self.graph = client.get_default_graph()
    self.graph.add_to_graph(self)

  def set_value(self, value):
    self.value = value

  def get_value(self):
    return self.value

  def forward(self):
    return self.value

  def grad(self):
    return 0


# Notice: ConstantOp is different from the one of TensorFlow
class ConstantOp(Op):
  def __init__(self, x, name=None):
    if name == None:
      self.name = "Constant"
    else:
      self.name = name

    # Notice, self.x should be Number or PlaceholerOp
    self.x = x

    self.graph = client.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return x

  def grad(self):
    return 0


class CoefficientOp(Op):
  def __init__(self, x, name=None):
    if name == None:
      self.name = "Coefficient"
    else:
      self.name = name

    self.x = x

    self.graph = client.get_default_graph()
    self.graph.add_to_graph(self)

  def set_x(self, x):
    self.x = x

  def get_x(self):
    return self.x

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return x

  def grad(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return x


class VariableOp(Op):
  def __init__(self, x, name=None):
    if name == None:
      self.name = "Variable"
    else:
      self.name = name

    self.x = x

    self.graph = client.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return x

  def grad(self):
    return 1


def test_VariableOp():
  x = 10
  variable = VariableOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class SquareOp(Op):
  def __init__(self, x, name=None):
    if name == None:
      self.name = "Square"
    else:
      self.name = name

    self.x = x

    self.graph = client.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x

    if os.environ.has_key("ENABLE_SWIG_OP"):
      result = swig.op.square(x)
    else:
      result = pow(x, 2)
    return result

  def grad(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return 2 * x


def test_SquareOp():
  x = 10
  variable = SquareOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class CubicOp(Op):
  def __init__(self, x, name=None):
    if name == None:
      self.name = "Cubic"
    else:
      self.name = name

    self.x = x

    self.graph = client.get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x

    if os.environ.has_key("ENABLE_SWIG_OP"):
      result = swig.op.cubic(x)
    else:
      result = math.pow(x, 3)
    return result

  def grad(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x

    if os.environ.has_key("ENABLE_SWIG_OP"):
      result = swig.op.multiple(3, swig.op.square(x))
    else:
      result = 3 * math.pow(x, 2)
    return result


def test_CubicOp():
  x = 10
  variable = CubicOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


def SigmoidOp(x):
  def __init__(self, x, name=None):
    if name == None:
      self.name = "Sigmoid"
    else:
      self.name = name

    self.x = x

    self.graph = client.get_default_graph()
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


# TODO: Composite op only supports linear combination
class AddOp(Op):
  def __init__(self, *ops):
    # TODO: Support user defined name in the parameter
    self.name = "Add"
    self.ops = ops
    self.name = None

    self.graph = client.get_default_graph()
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
  def __init__(self, *ops):
    self.name = "Multiple"
    self.ops = ops
    self.name = None

    self.graph = client.get_default_graph()
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
