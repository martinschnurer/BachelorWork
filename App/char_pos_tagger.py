import sys, os
import tensorflow as tf
import numpy as np
from getTokens import getTokens
from definedChars.charDict import CharDictionary
import random as rand
from datetime import datetime


DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/wiki-2014-02.ver')
VYZNAM_TOKENOV = os.path.join(os.path.dirname(__file__), 'tokeny_vyznam.csv')


charDict = CharDictionary()
tokens = getTokens()

is_ascii = lambda s: len(s) == len(s.encode())



num_hidden = 32


with tf.variable_scope('charTagger'):
    c_cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

'''
@function charTaggingGraph Create Graph, then returns multiplications
    which will be useful for merging with word_pos_tagging and mix into final
    layer
@args
    input Is used for data placeholder
    target Is used for learning
@returns MULTIPLICATION Multiplication will
'''
def charTaggingGraph(data):

    with tf.variable_scope('charTagger'):
        # [batch_size, sequence_length, size_of_vector]
        c_val, c_state = tf.nn.dynamic_rnn(
        	c_cell,
        	data,
        	dtype=tf.float32
        	)

    c_transposed = tf.transpose(c_val, [1, 0, 2])
    c_last = tf.gather(c_transposed, tf.shape(c_val)[1] - 1)

    c_weight = tf.Variable(tf.truncated_normal([num_hidden, 19]), name='c_weights')
    c_bias = tf.Variable(tf.constant(0.1), [19], name='c_biases')

    return tf.matmul(c_last, c_weight) + c_bias

def getMaxLength(sentence):
    maxLength = 0
    for word in sentence:
        if len(word) > maxLength:
            maxLength = len(word)
    return maxLength


def getZerosVector():
    return [0] * charDict.len


def getVectorFromWord(word):
    vec = []
    for char in word:
        char_vec = charDict.getVector(char)
        vec.append(char_vec)
    return vec


def getVectorFromSentence(sentence, isArray=True):
    vec = []
    maxLenWord = getMaxLength(sentence)
    if not isArray:
        sentence = sentence.split()
    for word in sentence:
        wordVec = getVectorFromWord(word)
        for _ in range(maxLenWord - len(wordVec)):
            wordVec.append(getZerosVector())
        vec.append(wordVec)
    return vec


def rand_seek(file):
    rand.seed(datetime.now())
    statinfo = os.stat(file.fileno())
    size = statinfo.st_size
    file.seek(rand.randint(0,size))
    try:
        next(file)
    except:
        file.seek(0)


def getNextBatch(file, dictObj, definedTokens, n=10, randomSeek=False):
    data_x, data_y = [],[]
    lines = []
    max_len = 0

    if randomSeek:
        rand_seek(file)

    while len(data_x) < n:
        line = file.readline()
        if line == '':
            file.seek(0)
            continue
        if '<' in line: # this is not valid line, so find next valid
            continue
        columns = line.split('\t')
        word = columns[0]
        lemma = columns[1]
        information = columns[2]
        if len(word) > max_len:
            max_len = len(word)

        tmp_list = []
        for char in word:
            tmp_list.append(dictObj.getVector(char))
        data_x.append(tmp_list)
        position = definedTokens[information[0]]['number'] - 1
        tmp_list = [0] * 19
        tmp_list[position] = 1
        data_y.append(tmp_list)

    for w in data_x:
        for i in range(max_len - len(w)):
            w.append([0] * dictObj.len)
    return data_x, data_y


if __name__ == "__main__":

    # dictionary containing methods for getting vector from one character
    # all characters are defined in file 'definedChars/chars'
    charDict = CharDictionary()

    # get ionary of all tokens
    tokens = getTokens()

    # Open dataset for reading words from file
    dataset = open(DATASET_PATH)



    if '--vyznamy_tokenov' in sys.argv:
        for token in tokens:
            print('{} {} {}'.format(token, tokens[token]['number'], tokens[token]['name']))
        exit(0)


    ########################   NEURAL NETWORK ####################################
    epochs = 150
    batch_size = 4
    num_hidden = 32
    # no_of_batches = int(len(data_x) / batch_size
    save_dir = "saved/{}hidden".format(num_hidden)
    save_path = "saved/{}hidden/model".format(num_hidden)


    # Create directory if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print("Creating Computational Graph")

    # [batch_size, sequence_length, size_of_vector]
    data = tf.placeholder(tf.float32, [None, None, charDict.len])
    target = tf.placeholder(tf.float32, [None, 19])

    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    val, state = tf.nn.dynamic_rnn(
    	cell,
    	data,
    	dtype=tf.float32
    	)

    transposed = tf.transpose(val, [1, 0, 2])
    last = tf.gather(transposed, tf.shape(val)[1] - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]), name='weights')
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]), name='biases')

    multiplication = tf.matmul(last, weight) + bias

    prediction = tf.nn.softmax(multiplication)
    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

    optimizer = tf.train.AdamOptimizer(0.01)
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    # -----------------------------------------

    if '--get-tag' in sys.argv:
        word = sys.argv[2]
        word_vec = getVectorFromWord(word, charDict)

        with tf.Session() as sess:
            try:
                saver.restore(sess, 'saved/{}hidden/model'.format(num_hidden))
            except:
                print("exception")
                sess.run(init)

            result = np.around(sess.run(prediction, {data: [word_vec]}), decimals=2)
            print(result)
            print(tokens)
        exit(1)

    #----------------------------------------
    if '--performance' in sys.argv:
        successful_marked = 0
        total = 10000
        rand_seek(dataset)
        with tf.Session() as sess:
            try:
                saver.restore(sess, 'saved/{}hidden/model'.format(num_hidden))
            except:
                print("exception")
                sess.run(init)
            inp, out = getNextBatch(dataset, charDict, tokens, n=total)

            for i in range(len(inp)):
                m_pred = sess.run(prediction, {data: [inp[i]]})
                if np.argmax(m_pred) == np.argmax(out[i]):
                    successful_marked = successful_marked + 1

            percents = (successful_marked * 100) / total
            print(percents)
            exit(1)
    #----------------------------------------





    with tf.Session() as sess:
        try:
            saver.restore(sess, 'saved/{}hidden/model'.format(num_hidden))
        except:
            print("exception")
            sess.run(init)

        for e in range(10000):

            # Get tensor, send to placeholder and minimize error
            inp, out = getNextBatch(dataset, charDict, tokens, n=batch_size)
            vysledok = sess.run(minimize, {data: inp, target: out })

            # Save model and try evaluate successfulness
            if e % 101 == 0 :
                saver.save(sess, 'saved/{}hidden/model'.format(num_hidden))
                tmp_cross_entr, tmp_err = sess.run([cross_entropy, error], {data: inp, target: out })
                print("Saving model", tmp_cross_entr, tmp_err)

                valid_x, valid_y = getNextBatch(dataset, charDict, tokens, 1, True)

                m_pred = sess.run([prediction], {data: valid_x, target: valid_y })

                # Print if prediction match target
                print("{} === {}".format(np.argmax(m_pred),np.argmax(valid_y)))
