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


data = tf.placeholder(dtype=tf.int32, shape=[None, None, 3])
max_numbers = tf.reduce_max(data, axis=2) + 1
sign = tf.sign(max_numbers)
suma = tf.reduce_sum(sign, axis=1)



init_op = tf.global_variables_initializer()


# maximum = tf.reduce_max(data,axis=-2)

data_x = [
    [
    [1,2,3],
    [4,8,7],
    [-1,-1,-1]
    ],
    [
    [1,2,3],
    [4,8,7],
    [5,8,1]
    ]
]


with tf.Session() as sess:
    sess.run(init_op)

    result = sess.run(data, feed_dict={data: data_x })
    print(result)
    result = sess.run(max_numbers, feed_dict={data: data_x })
    print(result)
    result = sess.run(sign, feed_dict={data: data_x })
    print(result)
    result = sess.run(suma, feed_dict={data: data_x })
    print(result)
