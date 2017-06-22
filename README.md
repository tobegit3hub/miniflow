# MiniFlow

## Introduction

It is the minimal implementation of [TensorFlow](https://github.com/tensorflow/tensorflow).

## Installation

```
pip install miniflow
```

## Usage

### Constant operations

Run with TensorFlow.

```
import tensorflow as tf

sess = tf.Session()
hello = tf.constant("Hello, TensorFlow!")
sess.run(hello)
# "Hello, TensorFlow!"
```

Run with MiniFlow.

```
import miniflow.miniflow as tf

sess = tf.Session()
hello = tf.constant("Hello, MiniFlow!")
sess.run(hello)
# "Hello, MiniFlow!"
```

### Basic operations

Run with TensorFlow.

```
import tensorflow as tf

sess = tf.Session()
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(sess.run(c))
# 42
```

Run with MiniFlow.

```
import miniflow.miniflow as tf

sess = tf.Session()
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(sess.run(c))
# 42
```

### Use placeholder

Run with TensorFlow.

```
import tensorflow as tf

sess = tf.Session()
a = tf.placeholder(tf.float32)
b = tf.constant(32.0)
c = tf.add(a, b)
print(sess.run(c, feed_dict={a: 10}))
print(sess.run(c, feed_dict={a.name: 10}))
# 42.0
```

Run with MiniFlow.

```
import miniflow.miniflow as tf

sess = tf.Session()
a = tf.placeholder(tf.float32)
b = tf.constant(32.0)
c = tf.add(a, b)
print(sess.run(c, feed_dict={a: 10.0}))
print(sess.run(c, feed_dict={a.name: 10.0}))
# 42.0
```
