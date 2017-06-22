# Swig

## Introduction

We can implement the operations in C++ which can be called in Python with [swig](http://www.swig.org/tutorial.html).

## Usage

```
import swig.op

swig.op.add(1.0, 2.0)
# 3.0

swig.op.multiple(3.0, 4.0)
# 12.0

swig.op.square(5.0)
# 25.0

swig.op.cubic(6.0)
# 216.0
```

## Implementation

### Example

```
swig -python example.i

gcc -c example.c example_wrap.c -I/usr/local/include/python2.1

ld -bundle -flat_namespace -undefined suppress example.o example_wrap.o -o _example.so
```

### Op

```
swig -python op.i

gcc -c op.c op_wrap.c -I/usr/local/include/python2.1

ld -bundle -flat_namespace -undefined suppress op.o op_wrap.o -o _op.so
```
