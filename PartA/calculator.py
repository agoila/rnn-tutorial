''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    descr: demonstrate the techniques and significance of persistent state by building a calculator 
    usage: Run `python calculator.py`.
           A prompt should appear.  Enter in a few numbers and observe the
           displayed stack grow.  Enter the 3-character command 'add' and 
           observe the computed sum.  Type 'exit' to close the calculator.
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 0.0. A program parameter:
STACK_SIZE = 10 

# 0.1. Let's build a simple desktop calculator by using a stack and adder.
Stack = tf.get_variable('stackvar', shape=(STACK_SIZE), dtype=tf.float32, initializer=tf.zeros_initializer()) 
Num = tf.placeholder(dtype=tf.float32) 

# 0.2. This function helps us specify the *order* in which nodes are executed:
def chain(*nodes):
    ''' Returns node that, when run, runs the given nodes *in sequence*.  The
        nodes are given as a list of 0-ary functions that return a node when
        called.  Contrast to `tf.group`, which allows (potentially
        nondeterministic) parallel computation. 
    '''
    if not nodes: return tf.no_op()
    with tf.control_dependencies([nodes[0]()]):
        return chain(*nodes[1:])

# 0.3   We use `chain` to define pushing and adding in terms of array slice
#       manipulations.  While `Stack` is not an array, we may refer to its
#       components through the standard Python slice notation:
Push= chain(
          lambda: tf.assign(Stack[1:], Stack[:-1]),
          lambda: tf.assign(Stack[0], Num),
      )
Add = chain(
          lambda: tf.assign(Stack[0], Stack[0]+Stack[1]),
          lambda: tf.assign(Stack[1:-1], Stack[2:]),
          lambda: tf.assign(Stack[-1], 0),
      )

###############################################################################
#                                 1. RUN GRAPH                                #
###############################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 1.0. Interact with the user to push numbers onto the stack and add them.
    #      See usage instructions in the header comment. 
    while True:
        user_command = input('>> ')
        if user_command=='exit': break
        elif user_command=='add': sess.run(Add)
        else: sess.run(Push, feed_dict={Num:float(user_command)})
        print(('Stack state:' + ' %d'*STACK_SIZE) % tuple(sess.run(Stack)))
