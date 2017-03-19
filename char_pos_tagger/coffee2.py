import tensorflow as tf
import numpy as np


"""
[
    [20 10 20 30 35 68 255]
    [50 31 20 18 1 255 0 5]
]
"""
def length(data):
    max_numbers = tf.reduce_max(data, axis=2) + 1
    sign = tf.sign(max_numbers)
    suma = tf.reduce_sum(sign, axis=1)
    return suma



# state = tf.Variable(state_, dtype=tf.float32)
# weights = tf.Variable(weights_, dtype=tf.float32)

#
# multiplication = tf.matmul(state, weights)
# softmax = tf.nn.softmax(multiplication)
#
#
# init_op = tf.global_variables_initializer()
#


epochs = 150
batch_size = 4
num_hidden = 6



print("Creating Computational Graph")

# [batch_size, sequence_length, size_of_vector]
data = tf.placeholder(tf.float32, [None, None, 5])
target = tf.placeholder(tf.float32, [None, 2])

cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

# SEQUENCE LENGTH POUZIT !
val, state = tf.nn.dynamic_rnn(
	cell,
	data,
	dtype=tf.float32
	)

# tf.shape je lepsie na pouzite --------------------------------->
# weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1]]), name='weights')
weight = tf.Variable(tf.truncated_normal([num_hidden, 2]), name='weights')
bias = tf.Variable(tf.constant(0.1, shape=[2], name='biases'))

multiplication = tf.matmul(tf.reshape(val,[3, num_hidden]) , weight)

prediction = tf.nn.softmax(multiplication)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
#
optimizer = tf.train.AdamOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)
#
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init = tf.global_variables_initializer()
# saver = tf.train.Saver()

word_ = [[
    [1,2,3,1,2],
    [1,1,1,1,1],
    [0,1,3,1,0]
]]

target_=[
    [0,1],
    [1,0],
    [0,1]
]


with tf.Session() as sess:
    sess.run(init)

    for a in range(100):
        vysledok, CE = sess.run([minimize, cross_entropy], {data: word_, target:target_} )
        print(CE)
    vysledok = sess.run(prediction, {data: word_} )
    print(np.around(vysledok, decimals=2))
