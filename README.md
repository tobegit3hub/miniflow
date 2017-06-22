# MiniFlow

## Introduction

MiniFlow aims to implement [TensorFlow](https://github.com/tensorflow/tensorflow) in minimal code.

* Support autograd for composite operations
* Support operations in C++ backend with swig
* Support Placeholer and feed_dict parameter
* Support same Graph/Session APIs as TensorFlow

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
print(sess.run(hello))
# "Hello, TensorFlow!"
```

Run with MiniFlow.

```
import miniflow.miniflow as tf

sess = tf.Session()
hello = tf.constant("Hello, MiniFlow!")
print(sess.run(hello))
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

## Release

Upload the package of [miniflow](https://pypi.python.org/pypi/miniflow/) in [pypi](https://pypi.python.org/pypi).

```
python ./setup.py sdist --format=gztar

twine upload dist/miniflow-x.x.x.tar.gz
```
