#!/usr/bin/env python

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

import sys
sys.path.append("../../")
sys.path.append("../../examples/miniflow_examples/")

import miniflow.miniflow as tf
import time

import linear_regression


def main():
  epoch_number = 100000
  print("Benchmark scenario: {}, epoch: {}".format("linear regression",
                                                   epoch_number))

  start_time = time.time()
  linear_regression.linear_regression(epoch_number, False)
  end_time = time.time()

  print("Run time(s): {}".format(end_time - start_time))  # Almost 13.787735939


if __name__ == "__main__":
  main()
