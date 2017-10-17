''' author: agoila
    changed: 2017-10-08
    created: 2017-10-05
    descr: generating script samples for LSTM based RNN
    usage: Run `python generate_lstm_sample.py`.
'''
import helper
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load parameters for generating a new script
_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, _, _, load_dir = helper.load_params_lstm()


###################################################################################################################
# Implement Generate Functions
# 
# Get Tensors
#
# Get tensors from loaded_graph using the function get_tensor_by_name(). Get the tensors using the following names:
# "input:0"
# "initial_state:0"
# "final_state:0"
# "probs:0"
#
# Since we are using name_scope(s), we need to add appropriate name_scope(s) prior to accessing its tensors.
# Return the tensors in the following tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
###################################################################################################################

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # Implement Function
    InputTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="input:0")
    InitialStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="lstm_cell/initial_state:0")
    FinalStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="lstm_cell/final_state:0")
    ProbsTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="model_out/probs:0")
    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)


##################################################################################
# Choose Word
#
# Implement the pick_word() function to select the next word using probabilities.
##################################################################################

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # Implement Function
    return np.random.choice(list(int_to_vocab.values()), 1, p=np.squeeze(probabilities))[0]


############################################################################################################
# Generate TV Script
#
# This will generate the TV script for you. Set gen_length to the length of TV script you want to generate.
############################################################################################################

gen_length = 200

# homer_simpson, moe_szyslak, or Barney_Gumble
# this is the word that you want your sentence to start with
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentence generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-gen_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[:, dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
