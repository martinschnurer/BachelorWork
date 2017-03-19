import tensorflow as tf
import numpy as np


lst =  [
            [
                [1,2,3,4],
                [4,5,6,7]
            ],
            [
                [7,8,9,10],
                [10,11,12,13]
            ],
            [
                [13,14,15,16],
                [16,17,18,19]
            ]
        ]

vec = tf.Variable(lst, dtype=tf.float32)
tps = tf.transpose(vec, perm=[2,1,0])


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(vec))
    print("")
    print(sess.run(tps))
