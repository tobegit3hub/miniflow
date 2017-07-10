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
"""
Alias MiniFlow APIs for TensorFlow APIs.

import miniflow as tf
tf.int32
tf.float32
tf.float64
tf.Graph
tf.Session
tf.Variable
tf.placeholder
tf.constant
tf.add
tf.minus
tf.multiple
tf.divide
tf.square
tf.global_variables_initializer
tf.local_variables_initialize
"""

# TODO: Need to import all after installation
from . import graph
from . import session
from . import ops
from . import optimizer
from . import train
from . import webhdfs

int32 = int
float32 = float
float64 = float

Graph = graph.Graph

Session = session.Session

Variable = ops.VariableOp
placeholder = ops.PlaceholderOp
constant = ops.ConstantOp
add = ops.AddOp
minus = ops.MinusOp
multiple = ops.MultipleOp
divide = ops.DivideOp
square = ops.SquareOp
global_variables_initializer = ops.GlobalVariablesInitializerOp
local_variables_initializer = ops.LocalVariablesInitializerOp
