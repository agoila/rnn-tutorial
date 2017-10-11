''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    descr: preview how to define and manipulate tensors
    usage: python tensor.py 
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 0.0. A program parameter for the size of the arrays to deal with:
SIZE = 10 

# 0.1. We can add array coefficients ``in parallel''!
A = tf.placeholder(tf.float32, shape=[SIZE])
B = tf.placeholder(tf.float32, shape=[SIZE])
Total = A + B 
#      The ability to express such computations of Tensorflow is another major
#      advantage.  Of course, whether or not the numeric additions actually
#      occur in parallel depends on the machine devices accessible to 
#      `tf.Session` below.  

# 0.2. Useful built-in operations include matrix multiplication
Matrix = tf.placeholder(tf.float32, shape=[SIZE, SIZE])
Product = tf.matmul(Matrix, tf.expand_dims(Total, 1))
#      Note the use of `tf.expand_dims` in order to turn a length-SIZE vector
#      into a SIZE-by-1 matrix.  This prepares the vector to be multiplied by
#      `Matrix`.

###############################################################################
#                                 1. RUN GRAPH                                #
###############################################################################

with tf.Session() as sess:
    # 1.0. Let us prepare some input arrays.  In real life, input arrays might
    #      represent images or text, but here, we will manually set:
    a = [1 for i in range(SIZE)] 
    b = [i for i in range(SIZE)] 

    # 1.1. Now run the graph.  Since we only query `Total`, which does not
    #      depend on `Matrix`, we do not need to specify `Matrix`'s value. 
    total = sess.run(Total, feed_dict={A:a, B:b})
    print('total:')
    print(total)

    # 1.2. Let's now test matrix multiplication.  We use a lower-triangular
    #      binary matrix.  This has the property of returning cumulative
    #      sums.  
    matrix = [[0 if i<j else 1 for j in range(SIZE)] for i in range(SIZE)]
    product = sess.run(Product, feed_dict={A:a, B:b, Matrix:matrix})
    print('cumulative:')
    print(product)
