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

import miniflow.miniflow as tf
import time


def main():
    print("Start benchmark")
    epoch_number = 100000

    sess = tf.Session()
    a = tf.constant(10.0)
    b = tf.constant(32.0)
    c = a + b

    start_time = time.time()
    for i in range(epoch_number):
        sess.run(c)
    end_time = time.time()

    print("Result: {}".format(end_time - start_time)) # Almost 13.787735939
    print("End of benchmark")

if __name__ == "__main__":
    main()