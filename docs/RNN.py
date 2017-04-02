#! /usr/bin/python3
import sys
import tensorflow as tf
from random import randint, shuffle
import numpy as np
import random


np.set_printoptions(threshold=np.nan)



# print(tf.Session().run(tf.reduce_max([[[1.5,5],[8,3]]],reduction_indices=2)))
# exit(0)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    print(used)
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    print(tf.Session().run(length))
    return length




train_input = []
train_output = []
ti  = []

num_of_samples = 250
for i in range(num_of_samples):
	temporary = []
    # pocet jednotiek
	random_number = randint(0, 20)
	for j in range(random_number):
		temporary.append([1])

	for j in range(20-random_number):
		temporary.append([0])

	shuffle(temporary)
	train_input.append(np.array(temporary))

	tmp_output = [0] * 20
	tmp_output[random_number - 1] = 1
	train_output.append(np.array(tmp_output))




train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

print("input and outputs generated...")

train_input = train_input
train_output = train_output

print(train_input)
print(train_output)
# expand_dim tensorflow
data = tf.placeholder(tf.float32, [None, 20,1])


target = tf.placeholder(tf.float32, [None, 21])

num_hidden = 3
cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)


# SEQUENCE LENGTH POUZIT !
val, state = tf.nn.dynamic_rnn(
	cell,
	data,
	dtype=tf.float32
	)


val = tf.transpose(val, [1, 0, 2])


# POZOR pri variabilnej dlzke retazca
# VIEM POUZIT tf.sequence_mask
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# tf.shape je lepsie na pouzite --------------------------------->
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]), name='weights')
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]), name='biases')

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    try:
        loaded_model = saver.restore(sess, './model.ckpt')
        print("Model nemohol byt nacitany")
    except:
        sess.run(init_op)

    save_path = saver.save(sess, "./model.ckpt")


    batch_size = 30
    no_of_batches = int(len(train_input)/batch_size)
    epoch = 10

    test_input = np.array([0] * 20)
    test_output = np.array([0] * 20)

    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
            ptr += batch_size
            print(sess.run(val, feed_dict={data: inp, target: out}))
            exit(0)
            sess.run(minimize, feed_dict={data: inp, target: out})
            vysledok = sess.run(last, feed_dict={data: inp, target: out})
        print(sess.run(error, feed_dict={data: inp, target: out}))
        print("Epoch - ", str(i))
        save_path = saver.save(sess, "./model.ckpt")

    print('-----------------')

    inp, out = [train_input[0]], [train_output[0]]
    print(inp)
    print(out)
    # print(sess.run(val,{data: inp, target: out}))
    print(sess.run(prediction,{data: inp, target: out}))

    valid_inp = np.array([[[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[1]]])
    print(np.around(sess.run(prediction, feed_dict={data: [valid_inp[0]]}), decimals=2))


    print(np.argmax(np.around(sess.run(prediction, feed_dict={data: [valid_inp[0]]}), decimals=2)))
