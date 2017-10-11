''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    credits: www.tensorflow.org/get_started/mnist/pros
             Thanks to Siraj Raval. His example code is
                 https://github.com/llSourcell/Tensorboard_demo/blob/master/simple.py
             and his corresponding YouTube video is
                https://www.youtube.com/watch?v=3bownM3L5zM
    descr: MNIST classifier with logging for use with Tensorboard 
           IDENTICAL to `fully_connected.py` except at 2.6 and 3.0. And train time is longer.
    usage: Run
               python logger.py &
           Then, while (or after) that is running, run
               tensorboard --logdir=logs --port=6006               
           And try viewing port localhost:6006 on a browser, e.g. 
               firefox http://127.0.1.1:6006
           To avoid complications, one might want to
               rm logs/*
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. LIST PROGRAM PARAMETERS                          #
###############################################################################

# 0.0. (Hyper)Parameters of Stochastic Gradient Descent:
TRAIN_TIME = 100000
BATCH_SIZE= 100
LEARNING_RATE = 0.5

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

# 2.0. Our current estimate of the true weights:
Weights = tf.get_variable('weightsvar', shape=[28*28,10], dtype=tf.float32, initializer=tf.zeros_initializer())
Biases = tf.get_variable('biasesvar', shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())

# 2.1. Placeholders for the data to which to fit `Weights` and `Biases`.  Note
#      that both `TrueInputs` and `TrueOutputs` are inputs to our graph:
TrueInputs = tf.placeholder(tf.float32, shape=[None, 28*28])
TrueOutputs= tf.placeholder(tf.float32, shape=[None, 10])

# 2.2. This line is our mathematical regression model.  Here, it is linear.  In
#      practice, one often uses complicated multiline expressions.  Here,
#      however, we interpret the regression outputs *not* as direct estimates of
#      `TrueOutputs` but instead as the estimated log probabilities (plus an
#      arbitrary and changing additive constant) of each potential true output
#      value.  In other words, our full regression model contains a ``softmax''   
#      nonlinearity.  For reasons of numerical stability, we model that 
#      nonlinearity in the loss function instead of here: 
PredictedOutputLogits = tf.matmul(TrueInputs, Weights) + Biases

# 2.3. Gradient Descent acts to minimize a differentiable loss (here Cross Entropy):
CrossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TrueOutputs, logits=PredictedOutputLogits))

# 2.4. Gradient Descent Step:
LearningRate = tf.placeholder(dtype=tf.float32)
Update = tf.train.GradientDescentOptimizer(LearningRate).minimize(CrossEntropyLoss)

# 2.5. Classification Diagnostics (how well did we do?).  Note the nice use of
#      `reduce_mean`. 
PredictionIsCorrect = tf.equal(tf.argmax(PredictedOutputLogits, 1), tf.argmax(TrueOutputs, 1))
Accuracy = tf.reduce_mean(tf.cast(PredictionIsCorrect, tf.float32))

# 2.6. SUMMARIES TO LOG AND VISUALIZE:
merged = tf.summary.merge([
             tf.summary.histogram('weights', Weights),
             tf.summary.histogram('biases', Biases),
             tf.summary.scalar('cross_entropy_loss', CrossEntropyLoss),
             ;tf.summary.scalar('accuracy', Accuracy),
         ] + [
             tf.summary.histogram('biases%d'%b, Biases[b]) for b in range(10)
         ])

###############################################################################
#                                 3. RUN GRAPH                                #
###############################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 3.0. CREATE WRITER:
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
    
    # 3.1. Repeatedly... 
    for i in range(TRAIN_TIME):
        #            ...sample a batch of training data:
        batch_inputs, batch_outputs = get_batch() 
        #            ...run the gradient descent update on that batch:
        sess.run(Update, feed_dict={TrueInputs:batch_inputs, TrueOutputs:batch_outputs, LearningRate:LEARNING_RATE}) 

        if i%100==0:
            batch_inputs, batch_outputs = get_batch() 
            summary = sess.run(merged, feed_dict={TrueInputs: batch_inputs, TrueOutputs: batch_outputs})
            writer.add_summary(summary, i)  # Write summary
            print(i)

    # 3.2. Report the final model's accuracy:
    accuracy = sess.run(Accuracy, feed_dict={TrueInputs: mnist.test.images, TrueOutputs: mnist.test.labels})
    print('Final accuracy: %.3f' % accuracy)
