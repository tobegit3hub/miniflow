#!/usr/bin/env python

import client
import ops

# Compatible for TensorFlow APIs
placeholder = ops.PlaceholderOp
constant = ops.ConstantOp
add = ops.AddOp
float32 = float
float64 = float
Session = client.Session


def main():
  # y = 3 * x**3 + 2 * x**2 + x + 10
  # y' = 9 * x**2 + 4 * x + 1
  print("Formula: {}".format("y = 3 * x**3 + 2 * x**2 + x + 10"))
  print("Formula gradient: {}".format("y' = 9 * x**2 + 4 * x + 1"))

  x = 1
  first_item = ops.MultipleOp(ops.CoefficientOp(3), ops.CubicOp(x))
  second_item = ops.MultipleOp(ops.CoefficientOp(2), ops.SquareOp(x))
  third_item = ops.VariableOp(x)
  forth_item = ops.ConstantOp(10)
  y = ops.AddOp(ops.AddOp(first_item, second_item), third_item, forth_item)

  # Should be "X: 1, forward: 16.0, grad: 14.0"
  print("X: {}, forward: {}, grad: {}".format(x, y.forward(), y.grad()))

  # Build the graph
  name_op_map = {}
  coefficient1 = ops.CoefficientOp(3)
  cubic1 = ops.CubicOp(x)
  coefficient2 = ops.CoefficientOp(2)
  square1 = ops.SquareOp(x)
  multiple1 = ops.MultipleOp(coefficient1, cubic1)
  multiple2 = ops.MultipleOp(coefficient2, square1)
  variable1 = ops.VariableOp(x)
  constant1 = ops.ConstantOp(10)
  add1 = ops.AddOp(multiple1, multiple2, variable1, constant1)
  add2 = ops.AddOp(multiple1, multiple2)

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
  x = ops.PlaceholderOp()
  coefficient1 = ops.CoefficientOp(3)
  cubic1 = ops.CubicOp(x)
  coefficient2 = ops.CoefficientOp(2)
  square1 = ops.SquareOp(x)
  multiple1 = ops.MultipleOp(coefficient1, cubic1)
  multiple2 = ops.MultipleOp(coefficient2, square1)
  variable1 = ops.VariableOp(x)
  constant1 = ops.ConstantOp(10)
  add1 = ops.AddOp(multiple1, multiple2, variable1, constant1)
  add2 = ops.AddOp(multiple1, multiple2)

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
  x = ops.PlaceholderOp()
  weights = ops.CoefficientOp(weights_value)
  cubic1 = ops.CubicOp(x)
  coefficient2 = ops.CoefficientOp(2)
  square1 = ops.SquareOp(x)
  multiple1 = ops.MultipleOp(weights, cubic1)
  multiple2 = ops.MultipleOp(coefficient2, square1)
  variable1 = ops.VariableOp(x)
  constant1 = ops.ConstantOp(10)
  add1 = ops.AddOp(multiple1, multiple2, variable1, constant1)

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
  def myfunction_python(x):
    return x**2 + x

  def myfunction(x):
    placeholer1 = ops.PlaceholderOp()
    # TODO: Set value after constructing the graph
    placeholer1.set_value(x)
    square1 = ops.SquareOp(placeholer1)
    variable1 = ops.VariableOp(placeholer1)
    add1 = ops.AddOp(square1, variable1)
    return add1.name

  myfunction_op_name = myfunction(10)
  myfunction_op = client.get_default_graph().name_op_map[myfunction_op_name]
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

  hello = ops.ConstantOp("Hello, TensorFlow! -- by MinialFlow")
  sess = client.Session()
  print(sess.run(hello))
  a = ops.ConstantOp(10)
  b = ops.ConstantOp(32)
  c = ops.AddOp(a, b)
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

  a = ops.PlaceholderOp()
  b = ops.ConstantOp(32)
  c = ops.AddOp(a, b)
  sess = Session()
  print(sess.run(c, feed_dict={a: 10}))
  print(sess.run(c, feed_dict={a.name: 10}))


if __name__ == "__main__":
  main()
