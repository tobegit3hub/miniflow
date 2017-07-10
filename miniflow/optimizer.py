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

from . import graph
from . import session
from . import ops


class OptimizerMinimizeOp(ops.Op):
  def __init__(self, optimizer, loss, name="OptimizerMinimize"):
    super(OptimizerMinimizeOp, self).__init__()

    self._optimizer = optimizer
    self._loss = loss

    self._graph = loss._graph
    self._graph.add_to_graph(self)

  def forward(self):
    variablename_grad_map = self._optimizer.compute_gradients(self._loss)
    self._optimizer.apply_gradients(variablename_grad_map)

  def grad(self):
    raise NotImplementedError


class Optimizer(object):
  def __init__(self, name="Optimizer"):
    self.name = name

  def minimize(self, loss):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

  def get_name(self):
    return self.name

  def set_name(self, name):
    self.name = name


class GradientDescentOptimizer(Optimizer):
  def __init__(self, learning_rate=0.01, name="GradientDescent"):
    super(GradientDescentOptimizer, self).__init__(name)

    self._learning_rate = learning_rate

    self._graph = graph.get_default_graph()

  def get_learning_rate(self):
    return self._learning_rate

  def set_learning_rate(self, learning_rate):
    # TODO: Check type of parameter
    self._learning_rate = learning_rate

  def get_graph(self):
    return self._graph

  def set_graph(self, graph):
    self._graph = graph

  def minimize(self, loss, global_step=None):
    return OptimizerMinimizeOp(self, loss)

  def compute_gradients(self, loss):

    variablename_variable_map = self._graph.get_trainable_variables_collection(
    )

    variablename_grad_map = {}
    for variable_name, variable in variablename_variable_map.items():
      grad = loss.grad(variable_name)
      variablename_grad_map[variable_name] = grad

    return variablename_grad_map

  def apply_gradients(self, variablename_grad_map):

    variablename_variable_map = self._graph.get_trainable_variables_collection(
    )

    for variable_name, variable in variablename_variable_map.items():
      grad = variablename_grad_map[variable_name]
      final_grad = self._learning_rate * grad
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
    super(GradientDescentOptimizer, self).__init__(name)
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._m = []
    self._v = []
