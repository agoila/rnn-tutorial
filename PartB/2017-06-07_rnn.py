''' author: samtenka
    changed: 2017-06-15 
    created: 2017-06-07
    credits: We learned much from
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    descr: Demonstrate RNN usage in TF via sequence-->sequence learning.
    usage: Run `python 2017-06-07_rnn.py`
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
def generate_dataset(seqs_per_batch, length, one_hot=False):
    ''' Return (X, Y), an input-output pair of batches. The array signatures:
            X:  seqs_per_batch x length x 1    floats 
            Y:  seqs_per_batch x length x 1    floats (x2 if one-hot; else x1)
        The rule is currently: output 1 when the cumulative sum of the inputs
        exceeds 0. This rule is deterministic and necessitates a long-term
        memory in the learner. 
    '''
    # produce X and Y:
    in_seqs = np.random.randn(seqs_per_batch, length, 1)
    modulator = np.expand_dims(np.cos(np.arange(length) * 6.283/2), 0)
    modulator = np.expand_dims(modulator, 2)
    modulator = np.tile(modulator, (seqs_per_batch, 1, 1))
    in_seqs_mod = np.multiply(in_seqs, modulator)
    #in_seqs_mod = in_seqs
    sums = np.cumsum(in_seqs_mod, axis=1) 
    out_labels = np.greater(sums, np.zeros_like(sums)).astype(np.int32)
     
    # convert Y to one-hot encoding:
    if one_hot:
        one_hots = np.zeros((seqs_per_batch, length, 2))
        rows = np.outer(np.arange(seqs_per_batch), np.ones(length)).astype(np.int32)
        cols = np.outer(np.ones(seqs_per_batch), np.arange(length)).astype(np.int32) 
        one_hots[rows, cols, out_labels[:,:,0]] = 1
        out_labels = one_hots

    return in_seqs, out_labels.astype(np.float32)

# 1. Create graph:
def RNN(in_seq, depth=2):
    ''' Build seq-to-seq RNN. For demonstration purposes, we stack 2 RNNs and
        use LSTM nodes. These features are overkill for our learning task.
    '''
    prev_layer = tf.unstack(in_seq, axis=1)
    # Stacked LSTMs:
    for i in range(depth):
        with tf.variable_scope('lay%d'%i):
            lstm_cell  = tf.contrib.rnn.BasicRNNCell(nb_hidden) # not actually LSTM node!
            prev_layer, _  = tf.contrib.rnn.static_rnn(lstm_cell, prev_layer, dtype=tf.float32)
    # Decode LSTM hidden states to output:  
    W = tf.Variable(tf.random_normal([nb_hidden, 1])) 
    WW = tf.tile(tf.expand_dims(W, 0), [seqs_per_batch, 1, 1])
    return tf.matmul(tf.stack(prev_layer, axis=1), WW)

# Model Subgraph:
x = tf.placeholder(tf.float32, [None, steps_per_seq, 1])
y = tf.placeholder(tf.float32, [None, steps_per_seq, 1])
pred = RNN(x)
pred = tf.nn.sigmoid(pred)
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
