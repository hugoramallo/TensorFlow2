import time
import numpy as np
import tensorflow as tf

#!mkdir data
#!mkdir data/ptb
#!wget -q -O data/ptb/reader.py https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork/labs/Week3/data/ptb/reader.py
#!cp data/ptb/reader.py .


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.io.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns: tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  rain_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.

    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)



#BUILDING MODEL LANGUAJE MODELING

#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
#The number of processing units (neurons) in the hidden layers
hidden_size_l1 = 256
hidden_size_l2 = 128
#The maximum number of epochs trained with the initial learning rate
max_epoch_decay_lr = 4
#The total number of epochs in training
max_epoch = 15
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 30
#The size of our vocabulary
vocab_size = 10000
embeding_vector_size= 200
#Training flag to separate training from testing
is_training = 1
#Data directory for our dataset
data_dir = "data/simple-examples/data/"

# Reads the data and separates it into training data, validation data and testing data
raw_data = ptb_raw_data(data_dir)
train_data, valid_data, test_data, vocab, word_to_id = raw_data
len(train_data)


def id_to_word(id_list):
    line = []
    for w in id_list:
        for word, wid in word_to_id.items():
            if wid == w:
                line.append(word)
    return line


print(id_to_word(train_data[0:100]))


itera = ptb_iterator(train_data, batch_size, num_steps)
first_touple = itera.__next__()
_input_data = first_touple[0]
_targets = first_touple[1]

_input_data.shape

_targets.shape

_input_data[0:3]

print(id_to_word(_input_data[0,:]))

#TENSOR 30X20X200

embedding_layer = tf.keras.layers.Embedding(vocab_size, embeding_vector_size,batch_input_shape=(batch_size, num_steps),trainable=True,name="embedding_vocab")

# Define where to get the data for our embeddings from
inputs = embedding_layer(_input_data)
inputs

#Constructing Recurrent Neural Networks
lstm_cell_l1 = tf.keras.layers.LSTMCell(hidden_size_l1)
lstm_cell_l2 = tf.keras.layers.LSTMCell(hidden_size_l2)
stacked_lstm = tf.keras.layers.StackedRNNCells([lstm_cell_l1, lstm_cell_l2])

#The input should be a Tensor of shape: [batch_size, max_time, embedding_vector_size], in our case it would be (30, 20, 200)
layer  =  tf.keras.layers.RNN(stacked_lstm,[batch_size, num_steps],return_state=False,stateful=True,trainable=True
init_state = tf.Variable(tf.zeros([batch_size,embeding_vector_size]),trainable=False)
layer.inital_state = init_state
layer.inital_state

#outputs
outputs = layer(inputs)
outputs

#Dense layer
dense = tf.keras.layers.Dense(vocab_size)
logits_outputs  = dense(outputs)
print("shape of the output from dense layer: ", logits_outputs.shape) #(batch_size, sequence_length, vocab_size)

#Activation layer
activation = tf.keras.layers.Activation('softmax')
output_words_prob = activation(logits_outputs)
print("shape of the output from the activation layer: ", output_words_prob.shape) #(batch_size, sequence_length, vocab_size)

print("The probability of observing words in t=0 to t=20", output_words_prob[0,0:num_steps])

#prediction
np.argmax(output_words_prob[0,0:num_steps], axis=1)
_targets[0]

def crossentropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
loss  = crossentropy(_targets, output_words_prob)

loss[0,:10]

#Now, we define cost as average of the losses:
cost = tf.reduce_sum(loss / batch_size)
cost

#training
# Create a variable for the learning rate
lr = tf.Variable(0.0, trainable=False)
optimizer = tf.keras.optimizers.SGD(lr=lr, clipnorm=max_grad_norm)

#assemble to build model
model = tf.keras.Sequential()
model.add(embedding_layer)
model.add(layer)
model.add(dense)
model.add(activation)
model.compile(loss=crossentropy, optimizer=optimizer)
model.summary()

#training variables
# Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
tvars = model.trainable_variablesç

#scope
[v.name for v in tvars]


#gradient
x = tf.constant(1.0)
y =  tf.constant(2.0)
with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    g.watch(y)
    func_test = 2 * x * x

var_grad = g.gradient(func_test, x) # Will compute to 10.0
print(var_grad)

var_grad = g.gradient(func_test, y) # Will compute to 3.0
print(var_grad)

with tf.GradientTape() as tape:
    # Forward pass.
    output_words_prob = model(_input_data)
    # Loss value for this batch.
    loss  = crossentropy(_targets, output_words_prob)
    cost = tf.reduce_sum(loss,axis=0) / batch_size

# Get gradients of loss wrt the trainable variables.
grad_t_list = tape.gradient(cost, tvars)

print(grad_t_list)

# Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
grads

# Create the training TensorFlow Operation through our optimizer
train_op = optimizer.apply_gradients(zip(grads, tvars))


class PTBModel(object):

    def __init__(self):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size_l1 = hidden_size_l1
        self.hidden_size_l2 = hidden_size_l2
        self.vocab_size = vocab_size
        self.embeding_vector_size = embeding_vector_size
        # Create a variable for the learning rate
        self._lr = 1.0

        ###############################################################################
        # Initializing the model using keras Sequential API  #
        ###############################################################################

        self._model = tf.keras.models.Sequential()

        ####################################################################
        # Creating the word embeddings layer and adding it to the sequence #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            self._embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embeding_vector_size,
                                                              batch_input_shape=(self.batch_size, self.num_steps),
                                                              trainable=True, name="embedding_vocab")  # [10000x200]
            self._model.add(self._embedding_layer)

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        # Create the LSTM Cells.
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument  of LSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        # LSTM cell processes one word at a time and computes probabilities of the possible continuations of the sentence.
        lstm_cell_l1 = tf.keras.layers.LSTMCell(hidden_size_l1)
        lstm_cell_l2 = tf.keras.layers.LSTMCell(hidden_size_l2)

        # By taking in the LSTM cells as parameters, the StackedRNNCells function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of stacked simple cells.
        stacked_lstm = tf.keras.layers.StackedRNNCells([lstm_cell_l1, lstm_cell_l2])

        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.
        # In step 2,  second word of each of the b sentences is input in parallel.
        # The parallelism is only for efficiency.
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly.
        # All the computations involving the words of all sentences in a batch at a given time step are done in parallel.

        ########################################################################################################
        # Instantiating our RNN model and setting stateful to True to feed forward the state to the next layer #
        ########################################################################################################

        self._RNNlayer = tf.keras.layers.RNN(stacked_lstm, [batch_size, num_steps], return_state=False, stateful=True,
                                             trainable=True)

        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = tf.Variable(tf.zeros([batch_size, embeding_vector_size]), trainable=False)
        self._RNNlayer.inital_state = self._initial_state

        ############################################
        # Adding RNN layer to keras sequential API #
        ############################################
        self._model.add(self._RNNlayer)

        # self._model.add(tf.keras.layers.LSTM(hidden_size_l1,return_sequences=True,stateful=True))
        # self._model.add(tf.keras.layers.LSTM(hidden_size_l2,return_sequences=True))
        ####################################################################################################
        # Instantiating a Dense layer that connects the output to the vocab_size  and adding layer to model#
        ####################################################################################################
        self._dense = tf.keras.layers.Dense(self.vocab_size)
        self._model.add(self._dense)

        ####################################################################################################
        # Adding softmax activation layer and deriving probability to each class and adding layer to model #
        ####################################################################################################
        self._activation = tf.keras.layers.Activation('softmax')
        self._model.add(self._activation)

        ##########################################################
        # Instantiating the stochastic gradient decent optimizer #
        ##########################################################
        self._optimizer = tf.keras.optimizers.SGD(lr=self._lr, clipnorm=max_grad_norm)

        ##############################################################################
        # Compiling and summarizing the model stacked using the keras sequential API #
        ##############################################################################
        self._model.compile(loss=self.crossentropy, optimizer=self._optimizer)
        self._model.summary()

        def crossentropy(self, y_true, y_pred):
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        def train_batch(self, _input_data, _targets):
            #################################################
            # Creating the Training Operation for our Model #
            #################################################
            # Create a variable for the learning rate
            self._lr = tf.Variable(0.0, trainable=False)
            # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
            tvars = self._model.trainable_variables
            # Define the gradient clipping threshold
            with tf.GradientTape() as tape:
                # Forward pass.
                output_words_prob = self._model(_input_data)
                # Loss value for this batch.
                loss = self.crossentropy(_targets, output_words_prob)
                # average across batch and reduce sum
                cost = tf.reduce_sum(loss / self.batch_size)
            # Get gradients of loss wrt the trainable variables.
            grad_t_list = tape.gradient(cost, tvars)
            # Define the gradient clipping threshold
            grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
            # Create the training TensorFlow Operation through our optimizer
            train_op = self._optimizer.apply_gradients(zip(grads, tvars))
            return cost

        def crossentropy(self, y_true, y_pred):
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        def train_batch(self, _input_data, _targets):
            #################################################
            # Creating the Training Operation for our Model #
            #################################################
            # Create a variable for the learning rate
            self._lr = tf.Variable(0.0, trainable=False)
            # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
            tvars = self._model.trainable_variables
            # Define the gradient clipping threshold
            with tf.GradientTape() as tape:
                # Forward pass.
                output_words_prob = self._model(_input_data)
                # Loss value for this batch.
                loss = self.crossentropy(_targets, output_words_prob)
                # average across batch and reduce sum
                cost = tf.reduce_sum(loss / self.batch_size)
            # Get gradients of loss wrt the trainable variables.
            grad_t_list = tape.gradient(cost, tvars)
            # Define the gradient clipping threshold
            grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
            # Create the training TensorFlow Operation through our optimizer
            train_op = self._optimizer.apply_gradients(zip(grads, tvars))
            return cost


########################################################################################################################
# run_one_epoch takes as parameters  the model instance, the data to be fed, training or testing mode and verbose info #
########################################################################################################################
def run_one_epoch(m, data, is_training=True, verbose=False):
    # Define the epoch size based on the length of the data, batch size and the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.
    iters = 0

    m._model.reset_states()

    # For each step and data point
    for step, (x, y) in enumerate(ptb_iterator(data, m.batch_size, m.num_steps)):

        # Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        # y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
        if is_training:
            loss = m.train_batch(x, y)
        else:
            loss = m.test_batch(x, y)

        # Add returned cost to costs (which keeps track of the total costs for this epoch)
        costs += loss

        # Add number of steps to iteration counter
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("Itr %d of %d, perplexity: %.3f speed: %.0f wps" % (
            step, epoch_size, np.exp(costs / iters), iters * m.batch_size / (time.time() - start_time)))

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return np.exp(costs / iters)

# Reads the data and separates it into training data, validation data and testing data
raw_data = ptb_raw_data(data_dir)
train_data, valid_data, test_data, _, _ = raw_data

# Instantiates the PTBModel class
m = PTBModel.instance()
K = tf.keras.backend
for i in range(max_epoch):
    # Define the decay for this epoch
    lr_decay = decay ** max(i - max_epoch_decay_lr, 0.0)
    dcr = learning_rate * lr_decay
    m._lr = dcr
    K.set_value(m._model.optimizer.learning_rate, m._lr)
    print("Epoch %d : Learning rate: %.3f" % (i + 1, m._model.optimizer.learning_rate))
    # Run the loop for this epoch in the training mode
    train_perplexity = run_one_epoch(m, train_data, is_training=True, verbose=True)
    print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))

    # Run the loop for this epoch in the validation mode
    valid_perplexity = run_one_epoch(m, valid_data, is_training=False, verbose=False)
    print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

# Run the loop in the testing mode to see how effective was our training
test_perplexity = run_one_epoch(m, test_data, is_training=False, verbose=False)
print("Test Perplexity: %.3f" % test_perplexity)




