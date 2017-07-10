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

import logging


class Graph(object):
  def __init__(self):
    self._name_op_map = {}

    self._trainable_variables_collection = {}

  def get_name_op_map(self):
    return self._name_op_map

  def get_trainable_variables_collection(self):
    return self._trainable_variables_collection

  def add_to_trainable_variables_collection(self, key, value):
    if key in self._trainable_variables_collection:
      logging.warning(
          "The key: {} exists in trainable_variables_collection".format(key))
    else:
      self._trainable_variables_collection[key] = value

  def get_unique_name(self, original_name):
    index = 0
    unique_name = "{}_{}".format(original_name, index)

    while unique_name in self._name_op_map.keys():
      index += 1
      unique_name = "{}_{}".format(original_name, index)

    return unique_name

  def add_to_graph(self, op):
    unique_name = self.get_unique_name(op.get_name())
    op.set_name(unique_name)
    self._name_op_map[op.get_name()] = op


# TODO: Make global variable for all packages
_default_graph = Graph()


def get_default_graph():
  global _default_graph
  if _default_graph == None:
    _default_graph = Graph()
  else:
    return _default_graph
