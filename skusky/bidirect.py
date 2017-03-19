#!/usr/bin/python3
import tensorflow as tf
import numpy as np
print("INFO: modules Tensorflow and Numpy imported...")


x = tf.placeholder(shape = [None, 3, 2], dtype=tf.float32, name='x')
y = tf.placeholder(shape = [None, 6, 6], dtype=tf.float32, name='y')


# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*62, 62]))
}
biases = {
    'out': tf.Variable(tf.random_normal([62]))
}


# Permuting batch_size and n_steps
x = tf.transpose(x, [1, 0, 2])
# Reshape to (n_steps*batch_size, n_input)
x = tf.reshape(x, [-1, 32])
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x = tf.split(x, 32, 0)

# Define lstm cells with tensorflow
# Forward direction cell
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(62, forget_bias=1.0)
# Backward direction cell
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(62, forget_bias=1.0)

outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

multi = tf.matmul(outputs[-1], weights['out']) + biases['out']





print("Graph successfuly created...")
