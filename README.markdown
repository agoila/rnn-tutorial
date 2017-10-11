# [MSAIL](http://msail.github.io) 2017 Tensorflow Tutorial
## by Ankit Goila, Uriah Israel, and Sam Tenka 
## presented on 2017-10-08 and 2017-10-15

This written tutorial assumes familiarity with Python (to the level of understanding
the recursive structure of `chain` in `calculator.py`), numpy (to the level of
understanding the broadcasting used in `linear_regression.py`'s generate section), and deep
learning (to the level of understanding why dropout regularizes).  In the
spoken tutorial, we will assume less and will be happy to address these preliminaries
as they come up.

## Section 0: Tensorflow Setup 
**Install**
`Python 3.x`
`numpy 1.x`
`tensorflow 1.x`.

**Try** the [Tensorflow MNIST tutorial](https://www.tensorflow.org/tutorials/mnist/pros/).
Do not worry if you do not understand this example at all: just ensure your
installation works and try to absorb the visual flavor of TF code. 

## Section 1: Tensorflow Basics
**Read and run**
`minimal.py`
`quadratic.py`
`tensors.py`.

**Exercise**: How does Tensorflow help with largescale computing? 

**Read and run**
`variables.py`
`calculator.py`
`linear_regression.py`.

**Exercise**: Experiment with the batch size and learning rate in `linear_regression.py`.
              How do they affect stochasticity of validation losses, convergence rate,
              and the quality of the asymptote?

**Exercise**: Modify `linear_regression.py` to `quadratic_regression.py`.  Do you observe 
              overfitting on the original, linear dataset?  Create an artificial dataset 
              on which you expect quadratic regression to outperform linear regression.
              Try your model on this new dataset.

**Exercise**: Modify `linear_regression.py` to use L1 instead L2 loss.  This means you
              will change both the loss computations and the gradient computations. 

## Section 2: Neural Networks

**Read and run**
`linear_regression_autodiff.py`
`fully_connected.py`

**Exercise**: During training, we report training accuracies.  Modify the code to report training _losses_ also.

**Exercise**: Add a hidden layer to the model in `fully_connected.py`.  Use a nonlinearity
              of your choice.  Can you improve the final test accuracy? 

## Section 3: Convolution

**Read and run**
`convolutional.py`

**Exercise**: List the hyperparameters you might want to vary in order to improve this model.

**Exercise**: How would you construct an image-to-image translating network?  Design an architecture.
              Hint: Google up ``deconvolutional layers''.

## Section 4: Tensorflow Workflow 
### Model Saving and Loading
**Read and run**
`saver.py`.

### Visualizing Network Behavior with Tensorboard 
**Read and run**
`logger.py`.  Then start and view tensorboard per `logger.py`'s header comments.

### Training on Flux 
Unfortunately, our Flux permissions have not yet been set up. 

## Section 5: Working with Sequences
### Recurrent Neural Networks

**Read and run**
`lstm-RNN.py`
`vanilla-RNN.py`

**Exercise**: Increase the depth of the RNN layers (vanilla and/or LSTM)
              and observe the loss and the network performance.
              Is there a clear correlation between depth and performance?

To generate text samples, run:
`generate_lstm_sample.py`
`generate_vanilla_sample.py`

Some code for preprocessing data is in:
`helper.py`

**Read and run**
`embedding-visualizer.ipynb`
