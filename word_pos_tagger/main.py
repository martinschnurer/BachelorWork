import sys, os
import tensorflow as tf
import numpy as np
import os
import random as rand
from datetime import datetime
from gensim.models import word2vec

def getTokens():
    tokens = {}
    with open('../tokeny_vyznam') as f:
        for line in f:
            arr = line.split('-')
            arr[-1] = arr[-1].strip()
            tokens[arr[0]] = { 'name':'', 'number' : -1}
            tokens[arr[0]]['name'] = arr[1]
            tokens[arr[0]]['number'] = int(arr[2])
    return tokens

def rand_seek(file):
    rand.seed(datetime.now())
    statinfo = os.stat(file.fileno())
    size = statinfo.st_size
    file.seek(rand.randint(0,size))
    try:
        next(file)
    except:
        file.seek(0)

def getSentence(file, tokens, randomSeek=False):
    sentence, target = [], []

    if randomSeek:
        rand_seek(file)

    # find nearest start of sentence, which is <s>

    line = '-'
    while True:
        line = file.readline().rstrip()
        if '<s>' in line:
            break

    line = ''
    while True:
        line = file.readline().rstrip()
        if '</s>' in line:
            break
        else:
            line = line.split('\t')
            sentence.append(line[0])
            target.append(tokens[line[2][0]]['number'])

    return sentence, target


dataset_file = open('../data/wiki-2014-02.ver')
tokens = getTokens()



num_hidden = 32
save_dir = "saved/{}hidden".format(num_hidden)
save_path = "saved/{}hidden/model".format(num_hidden)

# [batch_size, sequence_length, size_of_vector]
data = tf.placeholder(tf.float32, [1, None, 80])
target = tf.placeholder(tf.float32, [None, 19])

cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

# SEQUENCE LENGTH POUZIT !
val, state = tf.nn.dynamic_rnn(
	cell,
	data,
	dtype=tf.float32
	)

# tf.shape je lepsie na pouzite --------------------------------->
# weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1]]), name='weights')
weight = tf.Variable(tf.truncated_normal([num_hidden, 19]), name='weights')
bias = tf.Variable(tf.constant(0.1, shape=[19], name='biases'))

multiplication = tf.matmul(tf.reshape(val,[tf.shape(data)[1], num_hidden]) , weight)

prediction = tf.nn.softmax(multiplication)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


model = word2vec.Word2Vec.load_word2vec_format('../model/80cbow.bin', binary=True)


total_words = 0
correct_words = 0


with tf.Session() as sess:
    try:
        saver.restore(sess, 'saved/{}hidden/model'.format(num_hidden))
    except:
        print("exception")
        sess.run(init)


    for a in range(60000):
        words, trg = getSentence(dataset_file, tokens, randomSeek=False)
        # print(words, "\n", target)

        # MA TO BYT tensor [1,None,80] a target ONE_HOT [words_count, 19]
        data_x = [[]]
        data_y = []
        for abc in range(len(trg)):
            data_y.append([0] * 19)

        for w in words:
            if model.__contains__(w):
                data_x[0].append(model[w].tolist())
            else:
                data_x[0].append([0] * 80)

        for t in range(len(trg)):
            data_y[t][trg[t] - 1] = 1


        if a % 100 == 0:
            err, _ = sess.run([cross_entropy, minimize], {data:data_x, target:data_y})
            saver.save(sess, 'saved/{}hidden/model'.format(num_hidden))

            words, trg = getSentence(dataset_file, tokens, randomSeek=False)

            total_words = total_words + len(trg)

            data_x = [[]]
            data_y = []

            for abc in range(len(trg)):
                data_y.append([0] * 19)

            for w in words:
                if model.__contains__(w):
                    data_x[0].append(model[w].tolist())
                else:
                    data_x[0].append([0] * 80)

            for t in range(len(trg)):
                data_y[t][trg[t] - 1] = 1

            test_prediction = sess.run(prediction, {data:data_x})

            output_ = []
            for out_ in test_prediction:
                output_.append(np.argmax(out_) + 1)


            for a in range(len(trg)):
                if trg[a] == output_[a] :
                    correct_words = correct_words + 1

            print(correct_words * 100 / total_words)
