#!/usr/bin/env python

import tensorflow as tf

def main():
  a = tf.placeholder(tf.float32)
  b = tf.constant(32.0)
  c = tf.add(a, b)

  sess = tf.Session()
  print(sess.run(c, feed_dict={a: 10}))
  print(sess.run(c, feed_dict={a.name: 10}))

if __name__ == "__main__":
  main()
