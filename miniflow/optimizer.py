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

import graph
import session
import ops


class OptimizerMinimizeOp(ops.Op):
  def __init__(self, optimizer, loss, name="OptimizerMinimize"):
    super(OptimizerMinimizeOp, self).__init__()
    self.optimizer = optimizer
    self.loss = loss
    self.name = name

    self.graph = loss.graph
    self.graph.add_to_graph(self)

  def forward(self):
    variablename_grad_map = self.optimizer.compute_gradients(self.loss)
    self.optimizer.apply_gradients(variablename_grad_map)

  def grad(self):
    raise NotImplementedError


class Optimizer(object):
  def __init__(self, name=None):
    self.name = name

  def minimize(self, loss):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass


class GradientDescentOptimizer(Optimizer):
  def __init__(self, learning_rate=0.01, name="GradientDescent"):
    super(GradientDescentOptimizer, self).__init__(name)
    self.learning_rate = learning_rate

    self.graph = graph.get_default_graph()

  def minimize(self, loss, global_step=None):
    return OptimizerMinimizeOp(self, loss)

  def compute_gradients(self, loss):

    variablename_variable_map = self.graph.get_trainable_variables_collection()

    variablename_grad_map = {}
    for variable_name, variable in variablename_variable_map.iteritems():
      grad = loss.grad(variable_name)
      variablename_grad_map[variable_name] = grad

    return variablename_grad_map

  def apply_gradients(self, variablename_grad_map):

    variablename_variable_map = self.graph.get_trainable_variables_collection()

    for variable_name, variable in variablename_variable_map.iteritems():
      grad = variablename_grad_map[variable_name]
      final_grad = self.learning_rate * grad
      variable.set_value(variable.get_value() - final_grad)


class AdagradOptimizer(Optimizer):
  pass


class MomentumOptimizer(Optimizer):
  pass


class AdamOptimizer(Optimizer):
  def __init__(self,
               learning_rate=0.01,
               beta1=0.9,
               beta2=0.99,
               epsilon=0.01,
               name="Adam"):
    self.name = name
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.m = []
    self.v = []


def test_GradientDescentOptimizer():
  """
    opt = GradientDescentOptimizer(learning_rate=0.1)
    opt_op = opt.minimize(cost, var_list=<list of variables>)
  """
  pass
  """
  optimizer = GradientDescentOptimizer(learning_rate=0.1)
  print(optimizer)

  loss = "TODO"
  optimizer.minimize(loss)
  sess = session.Session()
  sess.run(optimizer)
  """
