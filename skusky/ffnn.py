#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import random as rnd

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

weights = tf.Variable(
	tf.random_uniform([1],-1,1)
	)
bias = tf.Variable(
	tf.random_uniform([1],-1,1)
	)

pred = x * weights + bias
cost = tf.reduce_mean(-tf.reduce_sum(pred * tf.log(y)))

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(cost)


x_data = []
y_data = []
for a in range(500):
	tmp_x = (int)(rnd.random() * 50)
	x_data.append([tmp_x])
	y_data.append([tmp_x * 2])


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100):
		sess.run(train, feed_dict={x: x_data, y: y_data})
		print(sess.run(cost, feed_dict={x: x_data, y: y_data}))

	print(sess.run(weights))
	print(sess.run(bias))
