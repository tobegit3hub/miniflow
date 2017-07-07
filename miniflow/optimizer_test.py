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

from optimizer import Optimizer
from optimizer import GradientDescentOptimizer


class OptimizerTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_init(self):
    foo = Optimizer()
    self.assertEqual(foo.__class__, Optimizer)

  def test_get_name(self):
    foo = Optimizer("FooOptimizer")
    self.assertEqual(foo.get_name(), "FooOptimizer")

  def test_get_name(self):
    foo = Optimizer()
    foo.set_name("FooOptimizer")
    self.assertEqual(foo.get_name(), "FooOptimizer")


class GradientDescentOptimizerTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_init(self):
    foo = GradientDescentOptimizer()
    self.assertEqual(foo.__class__, GradientDescentOptimizer)

  def test_compute_gradients(self):
    foo = GradientDescentOptimizer(learning_rate=0.1)
    """ TODO: Add more tests
    optimizer.minimize(loss)
    sess = session.Session()
    sess.run(optimizer)
    """


if __name__ == '__main__':
  unittest.main()
