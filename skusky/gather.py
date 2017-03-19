import tensorflow as tf
import numpy as np


lst = [
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [
            [11,12,13],
            [14,15,16],
            [17,18,19]
        ]
    ]

vec = tf.Variable(lst, dtype=tf.float32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run(tf.gather(vec, ))
    print(result)
