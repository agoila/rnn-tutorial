''' author: samtenka
    changed: 2017-06-15 
    created: 2017-06-15
    descr: Demonstrate RNN sequence generation
    usage: Run `python 2017-06-15_genrnn.py`
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

nb_hidden = 8
steps_per_seq = 16
seqs_per_batch = 256
batches_per_exec = 2024  
batches_per_print = 64 

# 0. Generate toy data:
def generate_dataset(seqs_per_batch, length):
    ''' Return array Y with signature:
            Y:  seqs_per_batch x length x 1
        The rule is currently: Y is the integral of some independent standard
        gaussian noise sequence.
    '''
    noise = np.random.randn(seqs_per_batch, length, 1)
    Y = np.cumsum(noise, axis=1) 
    return Y

# 1. Create graph:
def RNN(depth=2):
    ''' Build seq-to-seq RNN. For demonstration purposes, we stack 2 RNNs and
        use LSTM nodes. These features are overkill for our learning task.
    '''
    prev_layer = tf.unstack(in_seq, axis=1)
    # Stacked LSTMs:
    for i in range(depth):
        with tf.variable_scope('lay%d'%i):
            lstm_cell  = tf.contrib.rnn.BasicLSTMCell(nb_hidden)
            prev_layer, _  = tf.contrib.rnn.static_rnn(lstm_cell, prev_layer, dtype=tf.float32)
    # Decode LSTM hidden states to output:  
    W = tf.Variable(tf.random_normal([nb_hidden, 1])) 
    WW = tf.tile(tf.expand_dims(W, 0), [seqs_per_batch, 1, 1])
    return tf.matmul(tf.stack(prev_layer, axis=1), WW)

# Model Subgraph:
x = tf.placeholder(tf.float32, [None, steps_per_seq, 1])
y = tf.placeholder(tf.float32, [None, steps_per_seq, 1])
pred = RNN(x)
# Training Subgraph:
cost = tf.reduce_mean(tf.abs(y-pred))
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))
#accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), y), tf.float32)) 
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost) 

# 2. Train loop:
Ts = []
L1s = []
with tf.Session() as sess:
    # initialize:
    sess.run(tf.global_variables_initializer())
    for i in range(batches_per_exec):
        # train on fresh batch of data:  
        I, O = generate_dataset(seqs_per_batch, length=steps_per_seq)
        sess.run(optimizer, feed_dict={x:I, y:O}) 

        # print loss statistics every now and then: 
        if i % batches_per_print: continue
        c = sess.run(cost, feed_dict={x:I, y:O}) 
        #c, a = sess.run([cost, accu], feed_dict={x:I, y:O}) 
        #print('%4d\tXE: %.3f\tACC: %.3f' % (i, c, a))
        print('%4d\tL1: %.3f' % (i, c))
        Ts.append(i)
        L1s.append(c)

plt.plot(Ts, np.log(np.array(L1s)))
plt.savefig('moo.png')
