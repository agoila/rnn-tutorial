''' author: samtenka
    changed: 2017-06-15 
    created: 2017-06-14
    credits: We learned much from
        https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
    descr: Demonstrate *harmonic rnn* for seq2seq task
           [UNFINISHED]
    usage: Run `python 2017-06-14_hrnn.py`
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

steps_per_seq = 16
seqs_per_batch = 256
batches_per_exec = 4*2024 
batches_per_print = 64 

outdim = indim = 1 # for the cumulative-sum task below, MUST be equal 
hdim = 8 

# 0. Generate toy data:
def generate_dataset(one_hot=False):
    ''' Return (X, Y), an input-output pair of batches. The array signatures:
            X:  seqs_per_batch x steps_per_seq x indim      floats 
            Y:  seqs_per_batch x steps_per_seq x outdim     floats
        The rule is currently: output the sign of the cumulative sum of the inputs.
        (a 0 or a 1)
    '''
    # produce X and Y:
    in_seqs = np.random.randn(seqs_per_batch, steps_per_seq, 1)
    modulator = np.expand_dims(np.cos(np.arange(steps_per_seq) * 6.283/2), 0)
    modulator = np.expand_dims(modulator, 2)
    modulator = np.tile(modulator, (seqs_per_batch, 1, 1))
    in_seqs_mod = np.multiply(in_seqs, modulator)
    sums = np.cumsum(in_seqs_mod, axis=1) 
    out_labels = np.greater(sums, np.zeros_like(sums)).astype(np.float32)

    return in_seqs, out_labels

# 1. Create graph:

# complex numbers:
def complex_ones(shape):
    ''' Construct a complex weight variable close of all 1's'''
    init_real, init_imag = tf.ones(shape, dtype=tf.float32), tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(init_real), tf.Variable(init_imag)
def rand_complex(shape, sigma=1.0):
    ''' Construct a complex weight variable with unit gaussian distribution '''
    init_real, init_imag = [tf.random_normal(shape)*sigma for i in range(2)]
    return tf.Variable(init_real), tf.Variable(init_imag)
def mag_of(real, imag):
    ''' FILL IN '''
    return tf.sqrt(tf.square(real) + tf.square(imag))
def real_div(real, imag, scale):
    ''' FILL IN '''
    return tf.divide(real, scale), tf.divide(imag, scale)
def complex_assign(destr, desti, sourcer, sourcei):
    ''' FILL IN '''
    return tf.assign(destr, sourcer), tf.assign(desti, sourcei)
def complex_multiply(Ar, Ai, Br, Bi):
    ''' FILL IN '''
    Cr = tf.multiply(Ar, Br) - tf.multiply(Ai, Bi)
    Ci = tf.multiply(Ar, Bi) + tf.multiply(Ai, Br)
    return Cr, Ci
def complex_matmul(Ar, Ai, Br, Bi):
    ''' FILL IN '''
    Cr = tf.matmul(Ar, Br) - tf.matmul(Ai, Bi)
    Ci = tf.matmul(Ar, Bi) + tf.matmul(Ai, Br)
    return Cr, Ci
def complex_from_real(real):
    ''' FILL IN '''
    return real, tf.zeros_like(real) 
def real_from_complex(real, imag):
    ''' FILL IN
        Lossy!
    '''
    return real

def complex_add(A, B):
    ''' FILL IN '''
    Ar, Ai = A
    Br, Bi = B
    return Ar+Br, Ai+Bi

def HRNN_cell_weights(indim, hdim, outdim): 
    ''' Initialize HRNN cell weights and weight-clipper '''
    Ar, Ai = complex_ones([hdim]) # state-->state: pointwise mul
    Br, Bi = rand_complex([indim, hdim], sigma=1.0/indim**0.5) # in-->state
    Cr, Ci = rand_complex([hdim, outdim], sigma=1.0/hdim**0.5) # state-->out
    Sr, Si = rand_complex([indim, outdim], sigma=1.0/indim**0.5) # in-->out

    # TODO: collect clippers from various cells all together?
    A_mag = tf.maximum(mag_of(Ar, Ai), 1.0) # scale factor
    Ar_clipped, Ai_clipped = real_div(Ar, Ai, A_mag) 
    Ar_clipper, Ai_clipper = complex_assign(Ar, Ai, Ar_clipped, Ai_clipped)

    return Ar, Ai, Br, Bi, Cr, Ci, Sr, Si, Ar_clipper, Ai_clipper

def apply_HRNN_cell(inr, ini, old_stater, old_statei, Ar, Ai, Br, Bi, Cr, Ci, Sr, Si):
    ''' Construct interface for a harmonic recurrent cell: 
             _________________________
            |         OUTPUT          |
            |           ^             |
            |     <-- STATE <-- state |
            |           ^             |
            |         input           |
            |_________________________|
        The diagram above depicts this function's arguments in lowercase and
        its outputs in uppercase.
    ''' 
    new_stater, new_statei = complex_add(
                                 complex_multiply(Ar, Ai, old_stater, old_statei),
                                 complex_matmul(inr, ini, Br, Bi)
                             )
    # TODO: add bias to out?
    #       (e.g. out+=in OR out+=in*S)
    outr, outi = complex_matmul(new_stater, new_statei, Cr, Ci)
    return outr, outi, new_stater, new_statei

def HRNN(in_seqr, in_seqi, indim, hdim, outdim):
    ''' Construct HRNN interface.
        Here, in_seq is a list of tensors
        TODO: ELABORATE
    '''
    Ar, Ai, Br, Bi, Cr, Ci, Sr, Si, Ar_clipper, Ai_clipper = HRNN_cell_weights(indim, hdim, outdim)

    in_seqr, in_seqi = tf.unstack(in_seqr, axis=1), tf.unstack(in_seqi, axis=1) 

    stater, statei = rand_complex([hdim], sigma=0.01)
    out_seqr = []
    out_seqi = []
    for inr, ini in zip(in_seqr, in_seqi):
        outr, outi, stater, statei = apply_HRNN_cell(inr, ini, stater, statei, Ar, Ai, Br, Bi, Cr, Ci, Sr, Si)
        out_seqr.append(outr)
        out_seqi.append(outi)

    out_seqr, out_seqi = tf.stack(out_seqr, axis=1), tf.stack(out_seqi, axis=1)

    return out_seqr, out_seqi, Ar_clipper, Ai_clipper

def RNN(in_seq, depth=3):
    ''' Build seq-to-seq RNN. For demonstration purposes, we stack 2 RNNs and
        use LSTM nodes. These features are overkill for our learning task.
    '''
    prev_layer = tf.unstack(in_seq, axis=1)
    # Stacked LSTMs:
    for i in range(depth):
        with tf.variable_scope('lay%d'%i):
            lstm_cell  = tf.contrib.rnn.BasicRNNCell(hdim) # not actually LSTM node!
            prev_layer, _  = tf.contrib.rnn.static_rnn(lstm_cell, prev_layer, dtype=tf.float32)
    # Decode LSTM hidden states to output:  
    W = tf.Variable(tf.random_normal([hdim, outdim])) 
    WW = tf.tile(tf.expand_dims(W, 0), [seqs_per_batch, 1, 1])
    return tf.matmul(tf.stack(prev_layer, axis=1), WW)

#def stacked_HRNN(in_seq, indim, hdims, outdim):
# TODO: fill in 

in_seq = tf.placeholder(tf.float32, [seqs_per_batch, steps_per_seq, indim])
in_seqr, in_seqi = complex_from_real(in_seq)
out_seq = tf.placeholder(tf.float32, [seqs_per_batch, steps_per_seq, outdim])

clippers = []
#inter_seqr, inter_seqi, Ar_clipper, Ai_clipper = HRNN(in_seqr, in_seqi, indim, hdim, hdim)
#clippers += [Ar_clipper, Ai_clipper] 
#inter_seqr = tf.nn.sigmoid(inter_seqr)
#inter_seqi = tf.nn.sigmoid(inter_seqi)

pred_seqr, pred_seqi, Ar_clipper, Ai_clipper = HRNN(in_seqr, in_seqi, indim, hdim, outdim)
clippers += [Ar_clipper, Ai_clipper] 

pred_seq = real_from_complex(pred_seqr, pred_seqi)
pred_seq = RNN(pred_seq) 
pred_seq = tf.nn.sigmoid(pred_seq)

cost = tf.reduce_mean(tf.abs(pred_seq - out_seq))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

# 2. Train loop:
Ts = []
L1s = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(batches_per_exec+1):
        I, O = generate_dataset()
        sess.run(optimizer, feed_dict={in_seq:I, out_seq:O})
        sess.run(clippers)
        c = sess.run(cost, feed_dict={in_seq:I, out_seq:O})
        if i % batches_per_print: continue
        print('%4d\tL1: %.3f' % (i, c))
        Ts.append(i)
        L1s.append(c)
plt.plot(np.array(Ts), np.log(np.array(L1s)))
plt.savefig('goo.png')

print('Success!')
