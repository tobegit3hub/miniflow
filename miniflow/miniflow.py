#!/usr/bin/env python

import math
import numpy as np


class Graph(object):
  def __init__(self):
    self.name_op_map = {}

  def get_unique_name(self, original_name):
    index = 0
    unique_name = "{}_{}".format(original_name, index)

    while unique_name in self.name_op_map.keys():
      index += 1
      unique_name = "{}_{}".format(original_name, index)

    return unique_name

  def add_to_graph(self, op):
    op.name = self.get_unique_name(op.name)
    self.name_op_map[op.name] = op


# TODO: Make global variable for all packages
default_graph = Graph()


def get_default_graph():
  if default_graph == None:
    global default_graph
    default_graph = Graph()
  else:
    return default_graph


class Session(object):
  def __init__(self):
    pass

  def run(self, op, feed_dict=None, options=None):

    # Update the value of PlaceholerOp with feed_dict data
    name_op_map = op.graph.name_op_map

    if feed_dict != None:
      # Example: {"Placeholer_1": 10} or {PlaceholderOp: 10}
      for op_or_opname, value in feed_dict.items():
        if isinstance(op_or_opname, str):
          placeholder_op = name_op_map[op_or_opname]
        else:
          placeholder_op = op_or_opname
        if isinstance(placeholder_op, PlaceholderOp):
          placeholder_op.set_value(value)

    result = op.forward()
    return result


class Op(object):
  def __init__(self):
    pass

  def forward(self):
    pass

  def grad(self):
    pass


class PlaceholderOp(Op):
  def __init__(self, name=None):
    if name == None:
      self.name = "Placeholder"
    else:
      self.name = name

    self.value = None

    self.graph = get_default_graph()
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

    self.graph = get_default_graph()
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

    self.graph = get_default_graph()
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

    self.graph = get_default_graph()
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

    self.graph = get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return pow(x, 2)

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

    self.graph = get_default_graph()
    self.graph.add_to_graph(self)

  def forward(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return math.pow(x, 3)

  def grad(self):
    if isinstance(self.x, PlaceholderOp):
      x = self.x.get_value()
    else:
      x = self.x
    return 3 * math.pow(x, 2)


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

    self.graph = get_default_graph()
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

    self.graph = get_default_graph()
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

    self.graph = get_default_graph()
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


def main():
  x = 1

  # y = 3 * x**3 + 2 * x**2 + x + 10
  # y' = 9 * x**2 + 4 * x + 1
  print("Formula: {}".format("y = 3 * x**3 + 2 * x**2 + x + 10"))
  print("Formula gradient: {}".format("y' = 9 * x**2 + 4 * x + 1"))

  first_item = MultipleOp(CoefficientOp(3), CubicOp(x))
  second_item = MultipleOp(CoefficientOp(2), SquareOp(x))
  third_item = VariableOp(x)
  forth_item = ConstantOp(10)
  y = AddOp(AddOp(first_item, second_item), third_item, forth_item)

  # Should be "X: 1, forward: 16.0, grad: 14.0"
  print("X: {}, forward: {}, grad: {}".format(x, y.forward(), y.grad()))

  #y = 3 * CubicOp(x) + 2 * SquareOp(x) + VariableOp(x) + 10
  #print("X: {}, forward: {}, grad: {}".format(x, y.forward(), y.grad()))

  #y = CubicOp(x) * 3 + SquareOp(x) * 2 + VariableOp(x) + 10
  #print("X: {}, forward: {}, grad: {}".format(x, y.forward(), y.grad()))

  # Build the graph
  name_op_map = {}
  coefficient1 = CoefficientOp(3)
  cubic1 = CubicOp(x)
  coefficient2 = CoefficientOp(2)
  square1 = SquareOp(x)
  multiple1 = MultipleOp(coefficient1, cubic1)
  multiple2 = MultipleOp(coefficient2, square1)
  variable1 = VariableOp(x)
  constant1 = ConstantOp(10)
  add1 = AddOp(multiple1, multiple2, variable1, constant1)
  add2 = AddOp(multiple1, multiple2)

  name_op_map["coefficient1"] = coefficient1
  name_op_map["cubic1"] = cubic1
  name_op_map["coefficient2"] = coefficient2
  name_op_map["square1"] = square1
  name_op_map["multiple1"] = multiple1
  name_op_map["multiple2"] = multiple2
  name_op_map["variable1"] = variable1
  name_op_map["constant1"] = constant1
  name_op_map["add1"] = add1
  name_op_map["add2"] = add2

  print("add1 forward: {}, backward: {}".format(name_op_map["add1"].forward(),
                                                name_op_map["add1"].grad()))
  print("add2 forward: {}, backward: {}".format(name_op_map["add2"].forward(),
                                                name_op_map["add2"].grad()))

  # Build the grapah with placeholer
  name_op_map = {}
  x = PlaceholderOp()
  coefficient1 = CoefficientOp(3)
  cubic1 = CubicOp(x)
  coefficient2 = CoefficientOp(2)
  square1 = SquareOp(x)
  multiple1 = MultipleOp(coefficient1, cubic1)
  multiple2 = MultipleOp(coefficient2, square1)
  variable1 = VariableOp(x)
  constant1 = ConstantOp(10)
  add1 = AddOp(multiple1, multiple2, variable1, constant1)
  add2 = AddOp(multiple1, multiple2)

  name_op_map["coefficient1"] = coefficient1
  name_op_map["cubic1"] = cubic1
  name_op_map["coefficient2"] = coefficient2
  name_op_map["square1"] = square1
  name_op_map["multiple1"] = multiple1
  name_op_map["multiple2"] = multiple2
  name_op_map["variable1"] = variable1
  name_op_map["constant1"] = constant1
  name_op_map["add1"] = add1
  name_op_map["add2"] = add2

  x.set_value(1)
  print("Placeholder: {}, add1 forward: {}, backward: {}".format(
      1, name_op_map["add1"].forward(), name_op_map["add1"].grad()))
  print("Placeholder: {}, add2 forward: {}, backward: {}".format(
      1, name_op_map["add2"].forward(), name_op_map["add2"].grad()))

  x.set_value(2)
  print("Placeholder: {}, add1 forward: {}, backward: {}".format(
      2, name_op_map["add1"].forward(), name_op_map["add1"].grad()))
  print("Placeholder: {}, add2 forward: {}, backward: {}".format(
      2, name_op_map["add2"].forward(), name_op_map["add2"].grad()))

  # Automatically update weights with gradient
  learning_rate = 0.01
  epoch_number = 10
  label = 10
  weights_value = 3

  # y = weights * x**3 + 2 * x**2 + x + 10
  x = PlaceholderOp()
  weights = CoefficientOp(weights_value)
  cubic1 = CubicOp(x)
  coefficient2 = CoefficientOp(2)
  square1 = SquareOp(x)
  multiple1 = MultipleOp(weights, cubic1)
  multiple2 = MultipleOp(coefficient2, square1)
  variable1 = VariableOp(x)
  constant1 = ConstantOp(10)
  add1 = AddOp(multiple1, multiple2, variable1, constant1)

  for epoch_index in range(epoch_number):

    x.set_value(1)

    grad = add1.grad()
    weights_value -= learning_rate * grad
    weights.set_x(weights_value)

    predict = add1.forward()
    loss = predict - label
    print("Epoch: {}, loss: {}, grad: {}, weights: {}, predict: {}".format(
        epoch_index, loss, grad, weights_value, predict))

  # Run with default graph
  '''
  def tanh(x):                 # Define a function
    y = np.exp(-x)
    return (1.0 - y) / (1.0 + y)

  grad_tanh = grad(tanh)       # Obtain its gradient function
  grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
  '''

  def myfunction_python(x):
    return x**2 + x

  def myfunction(x):
    placeholer1 = PlaceholderOp()
    # TODO: Set value after constructing the graph
    placeholer1.set_value(x)
    square1 = SquareOp(placeholer1)
    variable1 = VariableOp(placeholer1)
    add1 = AddOp(square1, variable1)
    return add1.name

  myfunction_op_name = myfunction(10)
  myfunction_op = get_default_graph().name_op_map[myfunction_op_name]
  print("Run myfunction: {}, auto grad of myfucntion: {}".format(
      myfunction_op.forward(), myfunction_op.grad()))

  # Run with session
  ''' TensorFlow example
  hello = tf.constant('Hello, TensorFlow!')
  sess = tf.Session()
  print(sess.run(hello))
  a = tf.constant(10)
  b = tf.constant(32)
  c = tf.add(a, b)
  print(sess.run(c))
  '''

  hello = ConstantOp("Hello, TensorFlow! -- by MinialFlow")
  sess = Session()
  print(sess.run(hello))
  a = ConstantOp(10)
  b = ConstantOp(32)
  c = AddOp(a, b)
  print(sess.run(c))

  # Run with session and feed_dict
  '''
  
  a = tf.placeholder(tf.float32)
  b = tf.constant(32.0)
  c = tf.add(a, b)
  sess = tf.Session()
  print(sess.run(c, feed_dict={a: 10}))
  print(sess.run(c, feed_dict={a.name: 10}))
  '''

  a = PlaceholderOp()
  b = ConstantOp(32)
  c = AddOp(a, b)
  sess = Session()
  print(sess.run(c, feed_dict={a: 10}))
  print(sess.run(c, feed_dict={a.name: 10}))


if __name__ == "__main__":
  main()
