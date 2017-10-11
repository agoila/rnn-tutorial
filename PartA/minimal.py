''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    descr: illustrate near-minimal use of tensorflow 
    usage: python minimal.py 
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 0.0. Let us learn to add in Tensorflow.  Tensorflow allows us to define
#          GRAPHS --- like hardware blueprints
#          SESSIONS --- like usable hardware
#      First, we will specify in our blueprint that our adding machine should
#      have two inputs, i.e. two ``placeholders'':  
A = tf.placeholder(tf.int32)
B = tf.placeholder(tf.int32)
#      Then we need to wire those inputs through an adder to an output port:
Total = A + B
#      Think of `A`, `B`, and `Total` as blueprint symbols for nodes in a
#      circuit.  The above three lines define a GRAPH.

###############################################################################
#                                    1. RUN GRAPH                             #
###############################################################################

# 1.0. Now we create a SESSION in order to realize and use our GRAPH.  The
#      variable `sess` knows about the previously defined graph elements `A`,
#      `B`, and Total, even though we did not explicitly tell it.  This is
#      because `tf.placeholder`, `+` as overloaded for `A` and `B`, `tf.Session`
#      all implicitly communicate with global variables in the tensorflow
#      module.  Specifically, the GRAPH elements modify a global ``default
#      graph''.  `tf.Session` then tries to map the graph onto actual devices
#      and an actual, optimized program.  For instance, memory allocation
#      happens at this point.  ``Compilation'' of the graph also occurs here.
#
#      More complex projects may use multiple GRAPHS or multiple SESSIONS. 
with tf.Session() as sess:
    # 1.1. The SESSION is created.  Within this with-block, we may use our 
    #      `hardware` as we desire.  For instance, we can feed it inputs
    #      and query the result: 
    total = sess.run(Total, feed_dict={A:10, B:3})
    print('10 + 3 == %d' % total)

    # 1.2. Note that `Total` is a graph element while `total` is a numeric
    #      value.  In this case, `total` is an integer; later on, we will see
    #      cases where the numeric values are arrays of numbers.  `Total`, by
    #      contrast, has no numeric value.  It is a datastructure that models
    #      the flow of information we specified when we built the GRAPH.  It
    #      contains type information relevant to `total` (e.g. that `total`
    #      is a single integer rather than an array of floats etc).  When we
    #      run `sess.run`, the memory cells allocated to `A` and `B` get filled
    #      with 10 and 3, and the program constructed by `tf.Session` acts
    #      on those inputs until the memory cell allocated to `Total` is
    #      computed.  This tutorial will use upper- vs lower-case identifiers
    #      to distinguish graph elements from numeric values.

    # 1.3. We can run the graph again with different inputs...
    total = sess.run(Total, feed_dict={A:22, B:17})
    print('22 + 17 == %d' % total)
