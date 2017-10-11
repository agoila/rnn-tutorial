''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    descr: illustrate more complicated but memoryless use of tensorflow 
    usage: python quadratic.py 
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 0.0. The quadratic formula solves Ax^2 + Bx + C = 0 in terms of A, B, C:
A = tf.placeholder(tf.float32)
B = tf.placeholder(tf.float32)
C = tf.placeholder(tf.float32)

# 0.1. It is useful and common to define intermediate variables:
Disc = tf.sqrt(B*B - 4*A*C)
#      Unlike the case of a pure Python+Numpy solution, this intermediate
#      variable's numeric value need not ever be contained in a Python object.  
#      This is one major reason for using Tensorflow.

# 0.2. It is useful and common to define multiple ``outputs''.  Note:
#      Tensorflow does not distinguish output nodes as special; outputs  are
#      just whatever nodes whose values we directly care about.
XPlus = tf.divide(-B + Disc, 2*A)
XMinus = tf.divide(-B - Disc, 2*A)

###############################################################################
#                                1. RUN GRAPH                                 #
###############################################################################

with tf.Session() as sess:
    # 1.0. Compute sqrt(2) as the positive root of x^2-2 = 0: 
    x_plus = sess.run(XPlus, feed_dict={A:1, B:0, C:-2})
    print('\nsqrt(2) == %f' % x_plus)

    # 1.1. We may probe multiple nodes in the graph after a single execution: 
    x_plus, x_minus, disc = sess.run([XPlus, XMinus, Disc], feed_dict={A:1, B:-1, C:-1})
    print('        golden ratio == %f' % x_plus)
    print('another golden ratio == %f' % x_minus)
    print('discriminant         == %f' % disc)

    # 1.2. Math errors during graph execution lead to NaN:
    x_plus = sess.run(XPlus, feed_dict={A:1, B:0, C:1})
    print('\nsqrt(-1) == %f' % x_plus)
