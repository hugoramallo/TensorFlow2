import numpy as np
import tensorflow as tf


LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
state

lstm = tf.keras.layers.LSTM(LSTM_CELL_SIZE, return_sequences=True, return_state=True)

lstm.states=state

print(lstm.states)

#Batch size x time steps x features.
sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)

batch_size = 1
sentence_max_length = 1
n_features = 6

new_shape = (batch_size, sentence_max_length, n_features)

inputs = tf.constant(np.reshape(sample_input, new_shape), dtype = tf.float32)

#Now, we can pass the input to lstm_cell, and check the new state:
output, final_memory_state, final_carry_state = lstm(inputs)

print('Output : ', tf.shape(output))

print('Memory : ',tf.shape(final_memory_state))

print('Carry state : ',tf.shape(final_carry_state))

#stacked LSTM
cells = []

#first layer LTSM cell
LSTM_CELL_SIZE_1 = 4 #4 hidden nodes
cell1 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

#second layer
LSTM_CELL_SIZE_2 = 5 #5 hidden nodes
cell2 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

#multilayer LTSM
stacked_lstm =  tf.keras.layers.StackedRNNCells(cells)

#RNN from stacked_lstm
lstm_layer= tf.keras.layers.RNN(stacked_lstm ,return_sequences=True, return_state=True)

#Batch size x time steps x features.
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input

batch_size = 2
time_steps = 3
features = 6
new_shape = (batch_size, time_steps, features)

x = tf.constant(np.reshape(sample_input, new_shape), dtype = tf.float32)

#input network
output, final_memory_state, final_carry_state  = lstm_layer(x)

print('Output : ', tf.shape(output))

print('Memory : ',tf.shape(final_memory_state))

print('Carry state : ',tf.shape(final_carry_state))