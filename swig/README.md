# Swig

## Example

Refer to http://www.swig.org/tutorial.html .

```
swig -python example.i

gcc -c example.c example_wrap.c -I/usr/local/include/python2.1

ld -bundle -flat_namespace -undefined suppress example.o example_wrap.o -o _example.so
```

## Op

```
swig -python op.i

gcc -c op.c op_wrap.c -I/usr/local/include/python2.1 

ld -bundle -flat_namespace -undefined suppress op.o op_wrap.o -o _op.so
```
