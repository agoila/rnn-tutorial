''' author: agoila
    changed: 2017-10-15
    created: 2017-10-05
    descr: basic vanilla RNN for script generation
    usage: Run `python vanilla-RNN.py`.
'''
##############################################################################
# In this part of the tutorial, you'll generate your own Simpsons TV scripts 
# using basic vanilla RNNs. You'll be using part of the Simpsons 
# dataset of scripts from 27 seasons. The Neural Network you'll build 
# will generate a new TV script for a scene at Moe's Tavern.
###################################################################


###################################################################
# Import important libraries.
###################################################################

import tensorflow as tf
import helper
import numpy as np
import time
import random
from collections import Counter
from distutils.version import LooseVersion
import warnings
from tensorflow.contrib import seq2seq

###################################################################
# Step 0
#
# Making sure TensorFlow version is up-to-date.
# GPU would make training an RNN faster. 
# Recommended but not necessary.
###################################################################

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. It is recommended that you use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

###################################################################    
# Step 1
#
# Get the Data 
# The data is already provided for you in the folder "data/simpsons". 
# You'll be using a subset of the original dataset. It consists of 
# only the scenes in Moe's Tavern. This doesn't include other 
# versions of the tavern, like "Moe's Cavern", "Flaming Moe's", 
# "Uncle Moe's Family Feed-Bag", etc..
# The original dataset can be found here:
# https://www.kaggle.com/wcukierski/the-simpsons-by-the-data
###################################################################

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)

# Ignore the top notice, since we don't use it for analysing the data
text = text[81:]


########################################################################    
# Optional step
# Explore the Data 
# Play around with <view_sentence_range> values to view different parts 
# of the data. You can comment out the following lines to not have your program 
# print the output for every single run.
########################################################################

view_sentence_range = (0, 10)

print('Dataset Stats')
print('{} unique words'.format(len(set(text.split()))))
scenes = text.split('\n\n')
print('{} scenes'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('{:.2f} sentences per scene'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('{} lines'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('{} words per line'.format(np.average(word_count_sentence)))

print('\nThe sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

########################################################################  
# Step 2
#
# Implement Preprocessing Functions:
#
# The first thing to do to any dataset is preprocessing. We implement 
# the following preprocessing functions below:
#
#   Lookup Table
##  Tokenize Punctuation

########################################################################
# Lookup Table
#
# To create a word embedding (more on this later), we first need to 
# transform the words to ids. In this function, we create two dictionaries:
#
# Dictionary to go from the words to an id, we'll call vocab_to_int
# Dictionary to go from the id to word, we'll call int_to_vocab
#
# The function returns these dictionaries in the following tuple 
# (vocab_to_int, int_to_vocab)
########################################################################  

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    #Implement Function
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
    int_to_vocab = {vocab_to_int[word]: word for word in text}
    return (vocab_to_int, int_to_vocab)


###############################################################################
# Tokenize Punctuation
#
# We'll be splitting the script into a word array using spaces as delimiters. 
# However, punctuations like periods and exclamation marks make it hard for 
# the neural network to distinguish between the word "bye" and "bye!".
#
# We implement the function <token_lookup> to return a dict that will be 
# used to tokenize symbols like "!" into "||Exclamation_Mark||". 
#
# A dictionary has been created for the following symbols where the 
# symbol is the key and value is the token:
#
# Period ( . )
# Comma ( , )
# Quotation Mark ( " )
# Semicolon ( ; )
# Exclamation mark ( ! )
# Question mark ( ? )
# Left Parentheses ( ( )
# Right Parentheses ( ) )
# Dash ( -- )
# Return ( \n )
#
# This dictionary will be used to token the symbols and add the delimiter (space) 
# around them. This separates the symbols as their own word, making it easier 
# for the neural network to predict on the next word. 
# Special attention needs to be paid to tokens that may be confused as words.
# For eg. instead of using the token "dash", we try using something like "||dash||".
####################################################################################

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    #Implement Function
    token_dict = {'.': "||Period||", ',': "||Comma||", '"': "||Quotation_Mark||", \
                 ';': "||Semicolon||", '!': "||Exclamation_mark||", '?': "||Question_mark||", \
                 '(': "||Left_Parentheses||", ')': "||Right_Parentheses||", '--': "||Dash||", '\n': "||Return||"}
    
    return token_dict


###############################################################################
# Step 3
#
# Preprocess all the data and save it in a pickle file.
#
# We also load the useful information from the pickle file into
# variables to be later used by our model. 
###############################################################################

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


###############################################################################
# Step 4
#
# Neural Network Training Hyperparameters
#
# Tune the following parameters:
# 
# Set num_epochs to the number of epochs.
# Set batch_size to the batch size.
# Set rnn_size to the size of the RNNs.
# Set embed_dim to the size of the embedding.
# Set seq_length to the length of sequence.
# Set learning_rate to the learning rate.
# Set show_every_n_batches to the number of batches the neural network 
# should print progress.
#
# Exercise: Play around with these hyperparameters.
###############################################################################

# Number of Epochs
num_epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 128
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 16
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 10

# creating a save directory for the vanilla RNN model
save_dir_vanilla = './vanilla_save'


###################################################################################
# Step 5
#
# Creating Batches
#
# We implement <get_batches> to create batches of input and targets using int_text. 
# The batches should be a Numpy array with the shape 
# (number of batches, 2, batch size, sequence length).
#
# Each batch contains two elements:
# 
# The first element is a single batch of input with the shape [batch size, sequence length]
# The second element is a single batch of targets with the shape [batch size, sequence length]
#
# If we can't fill the last batch with enough data, we drop the last batch.
# For exmple, 
# get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2) 
# would return a Numpy array of the following:
#
#[
#  # First Batch
#  [
#    # Batch of Input
#    [[ 1  2], [ 7  8], [13 14]]
#    # Batch of targets
#    [[ 2  3], [ 8  9], [14 15]]
#  ]
#
#  # Second Batch
#  [
#    # Batch of Input
#    [[ 3  4], [ 9 10], [15 16]]
#    # Batch of targets
#    [[ 4  5], [10 11], [16 17]]
#  ]
#
#  # Third Batch
#  [
#    # Batch of Input
#    [[ 5  6], [11 12], [17 18]]
#    # Batch of targets
#    [[ 6  7], [12 13], [18  1]]
#  ]
#]
#
# Notice that the last target value in the last batch is the first input value of the 
# first batch. In this case, 1. This is a common technique used when creating sequence 
# batches, although it is rather unintuitive.
###################################################################################

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # Implement Function
    ## Get rid of spare data
    elements_per_batch = batch_size * seq_length
    num_batches = int(len(int_text) / elements_per_batch)
    
    # Drop the last few words to make only full batches
    xdata = np.array(int_text[: num_batches * elements_per_batch])
    ydata = np.array(int_text[1: num_batches * elements_per_batch + 1])
    
    # Reshape into batch_size rows & 
    # Split along axis=1 into 'num_batches' equal arrays
    x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)
    
    # Update last value of target to point to first input
    y_batches[-1][-1][-1] = x_batches[0][0][0]
    
    return np.array(list(zip(x_batches, y_batches)))


###################################################################################
# Step 6
#
# Build the model
#
# 1. Get inputs
# 2. Create embedding layer
# 3. Create a single RNN cell (vanilla RNN)
# 4. Build RNN (single layer sequence or multiple layers)
# 5. Add logits/probabilities/cost/optimizer calculations
#
###################################################################################

train_graph = tf.Graph()
with train_graph.as_default():
    
    # Define the vocab_size: 
    # our total vocabulary the network will work with
    vocab_size = len(int_to_vocab)

    
    #######################################################################
    # Create TF Placeholders for the Neural Network.
    # Input text placeholder named "input" using the TF Placeholder name parameter.
    # Targets placeholder
    # Learning Rate placeholder
    #######################################################################
    
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    lr = tf.placeholder(tf.float32, name="learning_rate")


    ####################################################################################
    # Word Embedding
    # Apply embedding to input_data using TensorFlow.
    # Using name_scope(s) helps build a cleaner model, easier to analyze with Tensorboard
    ####################################################################################
    
    with tf.name_scope("embedding_layer"):
        embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1), name="embedding")
        embed = tf.nn.embedding_lookup(embedding, inputs)


    ############################################################################    
    # Build RNN Cell and Initialize
    # Stack one or more BasicRNNCells (vanilla RNN) in a MultiRNNCell.
    # The RNN size should be set using rnn_size
    # Initalize Cell State using the MultiRNNCell's zero_state() function
    # Apply the name "initial_state" to the initial state using tf.identity()
    ############################################################################

    with tf.name_scope("vanilla_cell"):
        # Your basic RNN cell
        vanilla = tf.contrib.rnn.BasicRNNCell(rnn_size)
    
        # Stack up multiple RNN layers, for deep learning
        # Exercise: Change the number of layers from 1
        # to 2 or 3, and observe the results.        
        Cell = tf.contrib.rnn.MultiRNNCell([vanilla] * 1)
    
        input_data_shape = tf.shape(inputs)
    
        # Getting an initial state of all zeros
        initial_state = Cell.zero_state(input_data_shape[0], tf.float32)
        InitialState = tf.identity(initial_state, name="initial_state")
    
        ############################################################################  
        # You created an RNN Cell in above. Time to use the cell to create an RNN.
        # Build the RNN using the tf.nn.dynamic_rnn()
        # Apply the name "final_state" to the final state using tf.identity()
        # tf.nn.dynamic_rnn() performs a fully dynamic unrolling of inputs.
        ############################################################################  

        outputs, final_state = tf.nn.dynamic_rnn(Cell, embed, dtype=tf.float32)
        finalState = tf.identity(final_state, name="final_state")

        
    ############################################################################ 
    # Build the Neural Network
    #
    # Here, we apply a fully connected layer with a linear activation 
    # and vocab_size as the number of outputs.
    ############################################################################ 

    with tf.name_scope("model_out"):
        logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

    with tf.name_scope("model_cost"):
        # Loss function
        cost = seq2seq.sequence_loss(logits, targets, tf.ones([input_data_shape[0], input_data_shape[1]]))
        tf.summary.scalar("train_loss", cost)

    with tf.name_scope("train"):
        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


############################################################################
# Step 7
# 
# Train the network
#
# Train the neural network on the preprocessed data.
############################################################################

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('./logs_vanilla/1', sess.graph)
    saver = tf.train.Saver()
    iteration = 0

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {inputs: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                inputs: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
            
            iteration += 1

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))
                
            if iteration % 5 == 0:
                s = sess.run(summary_op, feed)
                file_writer.add_summary(s, iteration)
                

    # Save Model
    embed_mat = sess.run(embedding)
    saver.save(sess, './logs_vanilla/1/embedding.ckpt')
    saver.save(sess, save_dir_vanilla)
    print('Model Trained and Saved')
    
# Save parameters for checkpoint
helper.save_params_vanilla((seq_length, int_to_vocab, embed_mat, save_dir_vanilla))
