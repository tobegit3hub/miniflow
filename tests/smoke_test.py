#!/usr/bin/env python

import miniflow.miniflow as tf


def main():
  sess = tf.Session()

  hello = tf.constant("Hello, MiniFlow!")
  print(sess.run(hello))
  # "Hello, MiniFlow!"

  a = tf.constant(10)
  b = tf.constant(32)
  c = tf.add(a, b)
  print(sess.run(c))
  # 42

  sess = tf.Session()
  a = tf.placeholder(tf.float32)
  b = tf.constant(32.0)
  c = tf.add(a, b)
  print(sess.run(c, feed_dict={a: 10.0}))
  print(sess.run(c, feed_dict={a.name: 10.0}))
  # 42


if __name__ == "__main__":
  main()
