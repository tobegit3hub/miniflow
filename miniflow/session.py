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

import ops


class Session(object):
  """The session to run specified op from specified graph."""

  def __init__(self):
    pass

  def __enter__(self):
    """Support with statement."""
    return self

  def __exit__(self, type, value, trace):
    pass

  def run(self, op, feed_dict=None, options=None):

    # Update the value of PlaceholerOp with feed_dict data
    name_op_map = op.graph.get_name_op_map()

    if feed_dict != None:
      # Example: {"Placeholer_1": 10} or {PlaceholderOp: 10}
      for op_or_opname, value in feed_dict.items():
        if isinstance(op_or_opname, str):
          placeholder_op = name_op_map[op_or_opname]
        else:
          placeholder_op = op_or_opname

        if isinstance(placeholder_op, ops.PlaceholderOp):
          placeholder_op.set_value(value)

    result = op.forward()
    return result
