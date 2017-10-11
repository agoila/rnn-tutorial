import os
import pickle
import random
from collections import Counter
import numpy as np


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('data/preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('data/preprocess.p', mode='rb'))

def save_params_vanilla(params):
    """
    Save vanilla RNN parameters to file
    """
    pickle.dump(params, open('data/params_vanilla.p', 'wb'))


def save_params_lstm(params):
    """
    Save LSTM RNN parameters to file
    """
    pickle.dump(params, open('data/params_lstm.p', 'wb'))


def load_params_vanilla():
    """
    Load vanilla RNN parameters from file
    """
    return pickle.load(open('data/params_vanilla.p', mode='rb'))

def load_params_lstm():
    """
    Load LSTM RNN parameters from file
    """
    return pickle.load(open('data/params_lstm.p', mode='rb'))