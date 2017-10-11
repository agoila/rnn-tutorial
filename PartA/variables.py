''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    descr: demonstrate Tensorflow variables, i.e. persistent state
    usage: python variables.py 
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 0.0. Essential to flexible hardware and software is the idea of iteration.
#      But iteration only has value in the presence of persistent variables:
Count = tf.get_variable('countvar', shape=(), dtype=tf.int32, initializer=tf.zeros_initializer())
#      and the ability to assign:
Increment = tf.assign(Count, Count+1)
Reset = tf.assign(Count, 0)
#      Above, we gave `Count` the name 'countvar'.  Names are strings used to
#      aid in graph visualization and debugging.  They do not affect numerical
#      computation.

###############################################################################
#                                 1. RUN GRAPH                                #
###############################################################################

with tf.Session() as sess:
    # 1.0. Whenever one uses variables, run the following line:
    sess.run(tf.global_variables_initializer()) 

    # 1.1. The initializer `tf.zeros_initializer` starts Count at 0: 
    print('At initialization,  count==%d' % sess.run(Count))

    # 1.2. The initializer `tf.zeros_initializer` starts Count at 0: 
    sess.run(Increment)
    #      Aha! Unlike in previous examples without variables, the computation
    #      history (specifically, the fact that we ran Increment, affects this
    #      current computation:
    print('After incrementing, count==%d' % sess.run(Count))

    # 1.3 And so on... 
    sess.run(Increment)
    print('After incrementing, count==%d' % sess.run(Count))
    sess.run(Reset)
    print('After reset,        count==%d' % sess.run(Count))
