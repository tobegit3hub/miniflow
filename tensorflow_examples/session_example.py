#!/usr/bin/env python

import tensorflow as tf

def main():
  sess = tf.Session()
  a = tf.constant(10)
  b = tf.constant(32)
  c = tf.add(a, b)
  print(sess.run(c))

if __name__ == "__main__":
  main()
