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

import miniflow as tf
import miniflow


def linear_regression():
  epoch_number = 10
  learning_rate = 0.01
  train_features = [1.0, 2.0, 3.0, 4.0, 5.0]
  train_labels = [10.0, 20.0, 30.0, 40.0, 50.0]

  weights = tf.Variable(0.0)
  bias = tf.Variable(0.0)
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)

  predict = weights * x + bias
  loss = y - predict
  sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = sgd_optimizer.minimize(loss)

  with tf.Session() as sess:

    for epoch_index in range(epoch_number):
      # Take one sample from train dataset
      sample_number = len(train_features)
      train_feature = train_features[epoch_index % sample_number]
      train_label = train_labels[epoch_index % sample_number]

      # Update model variables and print loss
      sess.run(train_op, feed_dict={x: train_feature, y: train_label})
      loss_value = sess.run(loss, feed_dict={x: 1.0, y: 10.0})
      print("Epoch: {}, loss: {}, weight: {}, bias: {}".format(
          epoch_index, loss_value, sess.run(weights), sess.run(bias)))


def linear_regression_raw_api():
  epoch_number = 10
  learning_rate = 0.01
  train_features = [1.0, 2.0, 3.0, 4.0, 5.0]
  train_labels = [10.0, 20.0, 30.0, 40.0, 50.0]

  weights = miniflow.ops.VariableOp(0.0)
  bias = miniflow.ops.VariableOp(0.0)
  x = miniflow.ops.PlaceholderOp(float)
  y = miniflow.ops.PlaceholderOp(float)

  predict = weights * x + bias
  loss = y - predict
  sgd_optimizer = miniflow.optimizer.GradientDescentOptimizer(learning_rate)
  train_op = sgd_optimizer.minimize(loss)

  with miniflow.session.Session() as sess:

    for epoch_index in range(epoch_number):
      # Take one sample from train dataset
      sample_number = len(train_features)
      train_feature = train_features[epoch_index % sample_number]
      train_label = train_labels[epoch_index % sample_number]

      # Update model variables and print loss
      sess.run(train_op, feed_dict={x: train_feature, y: train_label})
      loss_value = sess.run(loss, feed_dict={x: 1.0, y: 10.0})
      print("Epoch: {}, loss: {}, weight: {}, bias: {}".format(
          epoch_index, loss_value, weights.get_value(), bias.get_value()))


def main():
  linear_regression()
  # linear_regression_raw_api()


if __name__ == "__main__":
  main()
