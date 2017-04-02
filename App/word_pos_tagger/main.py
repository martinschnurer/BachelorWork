import sys, os, signal
import tensorflow as tf
import numpy as np
import random as rand
from datetime import datetime
from gensim.models import word2vec

WORD_VEC_SIZE = 80
VYZNAM_TOKENOV = os.path.join(os.path.dirname(__file__), '../tokeny_vyznam.csv')
MODEL_CBOW = os.path.join(os.path.dirname(__file__), '../model/80cbow.bin')
TRAIN_DATASET_PATH = os.path.join(os.path.dirname(__file__), '../data/parsers/test/1')
VALIDATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), '../data/parsers/test/2')
TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), '../data/parsers/test/3')

#
# train_dataset = open('../data/parsers/test/1', 'r')
# validation_dataset = open('../data/parsers/test/2', 'r')
# test_dataset = open('../data/parsers/test/3', 'r')

model = 0


def getTokens():
    tokens = {}
    with open(VYZNAM_TOKENOV) as f:
        for line in f:
            arr = line.split(',')
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


def oneHot(which):
    which = which - 1
    vec = [0] * 19
    vec[which] = 1
    return vec


def getSentence(file, tokens, model, words_as_vectors=True, target_as_vectors=True):
    sentence = file.readline().rstrip().rstrip().split('\t')

    # print(sentence)

    if words_as_vectors:
        new_vector = []
        for w in sentence:
            new_vector.append(getWordEmbedding(w, model))
        sentence = new_vector

    targets = file.readline().rstrip().rstrip().split()
    # print(targets)
    targets = list(map(int, targets ))

    if target_as_vectors:
        targets = list(map(oneHot,targets)) # converts on one hot encoded vector



    return sentence, targets


def getWordEmbedding(word, model):
    if word in model:
        return model[word]
    else:
        return [0] * WORD_VEC_SIZE


"""
first argument = file descriptor for validate dataset
second argument = actual lowest recorded error on validation dataset

returns tuple (Bool, value) => bool if is overtrained, and value of actual error
"""
# isOvertrained(validation_dataset, tokens, max_error, sess, cross_entropy, model)
def isOvertrained(validate_fd, tokens, actual_lowest, sess, tf_cross_entropy, model, howmany=1000):
    error_sum = 0

    for i in range(howmany):
        wrd, trg = getSentence(validate_fd, tokens, model)
        wrd = [wrd]
        error_sum = error_sum + sess.run(tf_cross_entropy, {data: wrd, target: trg})

    if error_sum < actual_lowest:
        return False, error_sum
    return True, actual_lowest


train_dataset = open(TRAIN_DATASET_PATH, 'r')
validation_dataset = open(VALIDATION_DATASET_PATH, 'r')
test_dataset = open(TEST_DATASET_PATH, 'r')

# get tokens dictionary
tokens = getTokens()


num_hidden = 32
save_dir = "saved_with_validation/{}hidden".format(num_hidden)
save_path = "saved_with_validation/{}hidden/model".format(num_hidden)

# [ batch_size, sequence_length, size_of_vector ]
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

multiplication = tf.matmul(tf.reshape(val,[tf.shape(data)[1], num_hidden]), weight)

prediction = tf.nn.softmax(multiplication + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


model = word2vec.Word2Vec.load_word2vec_format(MODEL_CBOW, binary=True)


total_words = 0
correct_words = 0


with tf.Session() as sess:
    try:
        saver.restore(sess, 'saved_with_validation/{}hidden/model'.format(num_hidden))
    except:
        print("exception")
        sess.run(init)


    max_error = sys.maxsize
    for a in range(60000):
        words, trg = getSentence(train_dataset, tokens, model)
        words = [words]

        # print(words)
        # print(trg)

        sess.run(minimize, {data: words, target: trg})
        # print(some_err)

        if (a+1) % 1000 == 0:

            # error_sum = 0
            # can_be_saved = False
            # for i in range(1000):
            #     wrd, trg = getSentence(train_dataset, tokens, model)
            #     wrd = [wrd]
            #     ce_error = sess.run(cross_entropy, {data: wrd, target: trg})
            #     error_sum = error_sum + ce_error
            #
            # if error_sum < max_error:
            #     max_error = error_sum
            #     can_be_saved = True

            overtrained, max_error = isOvertrained(train_dataset, tokens, max_error, sess, cross_entropy, model)

            if not overtrained:
                print("model can be saved with error = {}".format(max_error))
                saver.save(sess, 'saved_with_validation/{}hidden/model'.format(num_hidden))
            else:
                print("overtrained detected - {}".format(max_error))

            # print(max_error)
