''' author: samtenka
    changed: 2017-10-08
    created: 2017-10-07
    credits: www.tensorflow.org/get_started/mnist/pros
    descr: convolutional classifier on MNIST, demonstrating saving 
           IDENTICAL to `convolutional.py` except in section 3 
    usage: Run `python convolutional.py`.
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. LIST PROGRAM PARAMETERS                          #
###############################################################################

# 0.0. (Hyper)Parameters of Stochastic Gradient Descent.  (Notice the smaller
#      learning rate):
TRAIN_TIME = 1000
BATCH_SIZE= 100
LEARNING_RATE = 0.01

###############################################################################
#                            1. READ DATASET                                  #
###############################################################################

# 0.0. MNIST is a classic image-classification dataset.  Its images are 28x28 
#      grayscale photographs of handwritten digits (0 through 9).  Note that
#      we load the labels in one-hot form.  This makes defining a loss function
#      easier: 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def get_batch(size=BATCH_SIZE):
    ''' Return `inputs` of shape (28*28,) and the corresponding
               `outputs` of shape (10,)
        randomly sampled from the full data. 
    '''
    inputs, outputs = mnist.train.next_batch(size)
    return inputs, outputs

###############################################################################
#                         2. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 2.0. Placeholders for the data to which to fit the model:
TrueInputs = tf.placeholder(tf.float32, shape=[None, 28*28])
TrueOutputs= tf.placeholder(tf.float32, shape=[None, 10])

# 2.1. MODEL HYPERPARAMETERS:
LearningRate = tf.placeholder(dtype=tf.float32)
KeepProb = tf.placeholder(tf.float32)

# 2.1. MODEL PARAMETERS (note the choice of initialization):
WeightsA= tf.get_variable('Wa', shape=[5, 5,  1, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
BiasesA = tf.get_variable('Ba', shape=[          32], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
WeightsB= tf.get_variable('Wb', shape=[3, 3, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
BiasesB = tf.get_variable('Bb', shape=[          64], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

WeightsC= tf.get_variable('Wc', shape=[5*5*64,  512], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
BiasesC = tf.get_variable('Bc', shape=[         512], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
WeightsD= tf.get_variable('Wd', shape=[   512,   10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
BiasesD = tf.get_variable('Bd', shape=[          10], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

# 2.2. BUILD CLASSIFIER:
def conv2d(x, W, stride=2, padding='VALID'):
    ''' Linear convolutional map  '''
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def lrelu(x):
    ''' leaky ReLU activation function '''
    return tf.maximum(0.1*x, x)

InputImages = tf.reshape(TrueInputs, [-1, 28, 28, 1])
HiddenLayerA = lrelu(conv2d(InputImages, WeightsA) + BiasesA)  # 12 x 12 x 32
HiddenLayerB = lrelu(conv2d(HiddenLayerA, WeightsB) + BiasesB) #  5 x  5 x 64

HiddenLayerC = lrelu(tf.matmul(tf.reshape(HiddenLayerB, [-1, 5*5*64]), WeightsC) + BiasesC)
Dropped = tf.nn.dropout(HiddenLayerC, KeepProb)
PredictedOutputLogits = tf.matmul(Dropped, WeightsD) + BiasesD

# 2.3. Gradient Descent acts to minimize a differentiable loss (here Cross Entropy):
CrossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TrueOutputs, logits=PredictedOutputLogits))

# 2.4. GRADIENT DESCENT STEP (note the change to ADAM):
Update = tf.train.AdamOptimizer(LearningRate).minimize(CrossEntropyLoss)

# 2.5. Classification Diagnostics (how well did we do?).  Note the nice use of
#      `reduce_mean`. 
PredictionIsCorrect = tf.equal(tf.argmax(PredictedOutputLogits, 1), tf.argmax(TrueOutputs, 1))
Accuracy = tf.reduce_mean(tf.cast(PredictionIsCorrect, tf.float32))

###############################################################################
#                                 3. RUN GRAPH                                #
###############################################################################

import glob

# 3.0. CREATE SAVER
saver = tf.train.Saver()
SAVE_PATH = 'checkpoints/conv.ckpt'

with tf.Session() as sess:
    # 3.1. LOAD OR INITIALIZE AS APPROPRIATE 
    if glob.glob(SAVE_PATH+'*'):
        print('Loading Model...')
        saver.restore(sess, SAVE_PATH)
    else:
        print('Initializing Model from scratch...')
        sess.run(tf.global_variables_initializer())
    
    # 3.2. Train the model...
    for i in range(TRAIN_TIME):
        batch_inputs, batch_outputs = get_batch() 
        sess.run(Update, feed_dict={TrueInputs:batch_inputs, TrueOutputs:batch_outputs, LearningRate:LEARNING_RATE, KeepProb:0.5}) 

        if i%50: continue
        batch_inputs, batch_outputs = get_batch() 
        # Note that at test time, KeepProb becomes 1...
        train_accuracy = sess.run(Accuracy, feed_dict={TrueInputs:batch_inputs, TrueOutputs:batch_outputs, KeepProb:1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

    # 3.3. Report the final model's accuracy:
    accuracy = sess.run(Accuracy, feed_dict={TrueInputs: mnist.test.images, TrueOutputs: mnist.test.labels, KeepProb:1.0})
    print('Final accuracy: %.3f' % accuracy)

    # 3.4. SAVE
    print('Saving Model...')
    saver.save(sess, SAVE_PATH)
