''' author: agoila
    changed: 2017-10-07
    created: 2017-10-05
    descr: generating script samples for LSTM based RNN
    usage: Run `python generate_lstm_sample.py`.
'''
import helper
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, _, _, load_dir = helper.load_params_lstm()

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # Implement Function
    InputTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="input:0")
    InitialStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="initial_state:0")
    FinalStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="final_state:0")
    ProbsTensor = tf.Graph.get_tensor_by_name(loaded_graph, name="probs:0")
    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # Implement Function
    return np.random.choice(list(int_to_vocab.values()), 1, p=np.squeeze(probabilities))[0]


gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
# this is the word that you want your sentence to start with
# play around with this a bit more to see how the RNN responds!
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    inputs, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {inputs: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-gen_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {inputs: dyn_input, initial_state: prev_state})

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