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

import unittest

import graph
import ops
from graph import Graph


class GlobalTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_get_default_graph(self):
    graph1 = graph.get_default_graph()
    graph2 = graph.get_default_graph()

    self.assertEqual(graph1.__class__, Graph)
    self.assertEqual(graph2.__class__, Graph)
    self.assertEqual(graph1, graph2)


class GraphTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_init(self):
    graph = Graph()
    self.assertEqual(graph.__class__, Graph)

  def test_get_name_op_map(self):
    graph = Graph()
    op = ops.Op()
    graph.add_to_graph(op)
    name_op_map = graph.get_name_op_map()
    self.assertEqual(name_op_map.keys()[0], op.name)
    self.assertEqual(name_op_map.values()[0], op)


if __name__ == '__main__':
  unittest.main()
