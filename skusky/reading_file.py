import tensorflow as tf

data_x = tf.placeholder(tf.float32, [None, 4, 3])
data_y = tf.placeholder(tf.float32, [None, 2, 2])

cell = tf.contrib.rnn.LSTMCell(num_units = 5)

output, state = tf.nn.dynamic_rnn(cell, data_x, dtype=tf.float32)


# pred = tf.matmul(data_x, output)


def getDataElement(file):
	words = file.readline().split(' ')
	lemmas = file.readline().split(' ')
	numbers = file.readline().split(' ')
	return words, lemmas, numbers

def test_dataset():
	file = open('../data/dataset')
	words, lemmas, numbers = getDataElement(file)
	print(numbers[:-1])
	file.close()



# SKUSKA = tf.Variable([[[1],[2]]], dtype=tf.float32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(tf.shape(SKUSKA)))
    # print(sess.run(output, feed_dict={data_x: [[[0,0,0],[0,1,0],[0,0,0],[0,0,0]]]}))
    print(sess.run(tf.one_hot([0,0,0,0,0], 255 ,1.,0.)))

# test_dataset()
