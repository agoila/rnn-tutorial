''' author: agoila
    changed: 2017-10-07
    created: 2017-10-05
    descr: basic vanilla RNN for script generation
    usage: Run `python vanilla-RNN.py`.
'''
import tensorflow as tf
import helper
import numpy as np
from collections import Counter
from distutils.version import LooseVersion
import warnings
from tensorflow.contrib import seq2seq

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. It is recommended that you use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)

# Ignore notice, since we don't use it for analysing the data
text = text[81:]


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

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    #Implement Function
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return (inputs, targets, learning_rate, keep_prob)

def get_init_cell(batch_size, rnn_size, keep_prob):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # Implement Function
    # Your basic RNN cell
    vanilla = tf.contrib.rnn.BasicRNNCell(rnn_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(vanilla, output_keep_prob=keep_prob)
    
    # Stack up multiple RNN layers, for deep learning
    # Exercise: Try changing 1 to 2 or 3
    Cell = tf.contrib.rnn.MultiRNNCell([drop] * 1)
    
    # Getting an initial state of all zeros
    initial_state = Cell.zero_state(batch_size, tf.float32)
    InitialState = tf.identity(initial_state, name="initial_state")
    
    return (Cell, InitialState)

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # Implement Function
    embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    finalState = tf.identity(final_state, name="final_state")
    return (outputs, finalState)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # Implement Function
    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    return (logits, final_state)

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

# Number of Epochs
num_epochs = 60
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
# Set keep_prob
keep_prob = 1.0

save_dir_vanilla = './vanilla_save'

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr, keep_pr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, keep_prob)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate,
                keep_pr: keep_prob}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir_vanilla)
    print('Model Trained and Saved')
    
# Save parameters for checkpoint
helper.save_params_vanilla((seq_length, save_dir_vanilla))