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

from ops import Op
from ops import VariableOp
from ops import PlaceholderOp
from ops import SquareOp


class OpTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_init(self):
    foo = Op()
    self.assertEqual(foo.__class__, Op)

  def test_forward(self):
    foo = Op()

    with self.assertRaises(NotImplementedError):
      foo.forward()

  def test_grad(self):
    foo = Op()

    with self.assertRaises(NotImplementedError):
      foo.grad()


class SquareOpTest(unittest.TestCase):
  def test_SquareOp(self):
    w = VariableOp(10)
    b = VariableOp(20)
    x = PlaceholderOp(float)
    x.set_value(2.0)
    y = PlaceholderOp(float)
    y.set_value(3.0)
    """
    loss = SquareOp(y - (w * x + b))
    print("w: {}, forward: {}, grad: {}".format(w.get_value(),
                                                loss.forward(),
                                                loss.grad(w.get_name())))  # 148.0
    print("b: {}, forward: {}, grad: {}".format(b.get_value(),
                                                loss.forward(),
                                                loss.grad(b.get_name())))  # 74.0
    """


if __name__ == '__main__':
  unittest.main()
