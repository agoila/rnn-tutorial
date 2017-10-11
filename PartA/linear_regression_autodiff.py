''' author: samtenka
    changed: 2017-10-06
    created: 2017-10-06
    descr: linear regression via automatic differentiation   
           IDENTICAL to `linear_regression.py` except in Section 2
    usage: Run `python linear_regression_autodiff.py`.
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. LIST PROGRAM PARAMETERS                          #
###############################################################################

# 0.0. Parameters controlling how we generate our toy dataset:
DIMENSION = 2
DATA_SIZE = 1000

# 0.1. (Hyper)Parameters of Stochastic Gradient Descent:
TRAIN_TIME = 10
BATCH_SIZE= 10
LEARNING_RATE = 0.001

###############################################################################
#                            1. GENERATE DATASET                              #
###############################################################################

# 1.0. The true data will be an array of input vectors and a corresponding 
#      array of output numbers.  Each input vector has an output number 
#      given by a linear map with weights `true_weights` plus some additive
#      noise.  In practice, we would not generate an artificial dataset but
#      instead use real-world data.
true_weights = np.arange(DIMENSION)
full_inputs = np.random.randn(DATA_SIZE, DIMENSION) * 10 
full_outputs = np.dot(full_inputs, true_weights) + np.random.randn(DATA_SIZE) 

# 1.1. We will estimate the true weights by looking at a small random batch of
#      true data at a time.  This helper function returns those batches:
def get_batch(size=BATCH_SIZE):
    ''' Return `inputs` of shape (size, DIMENSION) and the corresponding
               `outputs` of shape (size,)
        randomly sampled from the full data. 
    '''
    indices = np.random.choice(DATA_SIZE, size)
    inputs = full_inputs[indices]
    outputs = full_outputs[indices]
    return inputs, outputs

###############################################################################
#                         2. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 2.0. Our current estimate of the true weights:
Weights = tf.get_variable('weightvar', shape=[DIMENSION], dtype=tf.float32, initializer=tf.zeros_initializer()) 

# 2.1. Placeholders for the regression data to which to fit `Weights`.  Note
#      that both `TrueInputs` and `TrueOutputs` are inputs to our graph:
TrueInputs = tf.placeholder(dtype=tf.float32, shape=(None, DIMENSION)) 
TrueOutputs = tf.placeholder(dtype=tf.float32, shape=(None)) 

# 2.2. This line is our mathematical regression model.  Here, it is linear.  In
#      practice, one often uses complicated multiline expressions.  Those
#      expressions are often neural networks.  In other words, this is the line
#      that you would replace with a neural network in order to perform better
#      on complicated date:
PredictedOutputs = tf.reshape(tf.matmul(TrueInputs, tf.expand_dims(Weights, 1)), shape=[-1])

# 2.3. Gradient Descent acts to minimize a differentiable loss (here MSE):
Loss = tf.reduce_mean(tf.square(PredictedOutputs - TrueOutputs))

# 2.4. Thanks to AUTOMATIC DIFFERENTIATION, we do not need to compute gradients manually:
#GradPredictedOutputs = 2 * (PredictedOutputs - TrueOutputs)
#GradWeights = tf.reduce_mean(tf.multiply(TrueInputs, tf.expand_dims(GradPredictedOutputs, 1)), axis=0)

# 2.5. We also change the Gradient Descent Step:
LearningRate = tf.placeholder(dtype=tf.float32)
#      Instead of manual gradient updates, we use a Tensorflow Optimizer:
#Update = tf.assign(Weights, Weights - LearningRate*GradWeights)
Update = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)  


###############################################################################
#                                 3. RUN GRAPH                                #
###############################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 3.0. Repeatedly... 
    for i in range(TRAIN_TIME):
        #            ...sample a batch of training data:
        batch_inputs, batch_outputs = get_batch() 
        #            ...run the gradient descent update on that batch:
        sess.run(Update, feed_dict={TrueInputs:batch_inputs, TrueOutputs:batch_outputs, LearningRate:LEARNING_RATE}) 
        #            ...sample another batch (here from the training data, but
        #               in practice, from validation data separate from the
        #               training data):
        batch_inputs, batch_outputs = get_batch() 
        #            ...then compute and report the loss on that batch:
        loss = sess.run(Loss, feed_dict={TrueInputs:batch_inputs, TrueOutputs:batch_outputs})
        print('\tLoss on batch %d: %.2f' % (i, loss))

    # 3.1. The estimated and true weights should roughly agree: 
    print('True weights: %s' % str(tuple(true_weights)))
    print('Estimated weights: %s' % str(tuple(sess.run(Weights))))
