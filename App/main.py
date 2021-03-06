import os, sys
import tensorflow as tf
import numpy as np
import getTokens as tokensGetter
from termcolor import colored

import char_pos_tagger as charTagger
import word_pos_tagger as wordGraph

from dataset import MODEL_CBOW, Dataset
from gensim.models import word2vec
from definedChars.charDict import CharDictionary
from OvertrainDetector import OvertrainDetector

# SLUZI pri kontrolovani overlearningu
MIN_VALIDATION_ERROR = sys.maxsize

TOTAL_WORDS = 0
MISTAKEN_WORDS = 0

print('--------------------------')
print('Modules loaded successfuly')

model = word2vec.Word2Vec.load_word2vec_format(MODEL_CBOW, binary=True)
charDict = CharDictionary()
DS = Dataset(model=model)

validation_file = open('data/parsers/test/2', 'r')
validation_DS = Dataset(model=model, file_dataset=validation_file)



# VALIDATION_DS ERROR
validation_dataset_error = tf.Variable(sys.maxsize, dtype=tf.float64)

# [None, None, charDict.len]
# None - pocet slov, neviem
# None - maximalny pocet pismen -> tiez neviem aky bude najvacsi pocet pismen vo vete
# Ale viem aky bude embedding
char_inp = tf.placeholder(shape=[None, None, charDict.len], dtype=tf.float32, name='char_inp')
char_multiplication = charTagger.charTaggingGraph(char_inp)

word_inp = tf.placeholder(shape=[1, None, 80], dtype=tf.float32, name='word_inp')
word_multiplication = wordGraph.wordPosTagger(word_inp)

target = tf.placeholder(tf.float32, [None, 19], name='target')


final_addition = word_multiplication + char_multiplication
prediction = tf.nn.softmax(final_addition)
# shapeOfResult = tf.shape(softmax)

cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


prediction_max_arg = tf.argmax(prediction, axis=1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def isOvertrained(sess, tf_cross_entropy, model_ref, validation_DS, tf_validation_error):
    error_sum = 0
    error_now = sess.run(tf_validation_error)

    for i in range(1000):
        _charInput = validation_DS.getCharGraphInput()
        _wordInput = validation_DS.getWordGraphInput()
        _target = validation_DS.getTarget()

        error_sum += sess.run(tf_cross_entropy, {
                                        word_inp : _wordInput,
                                        char_inp : _charInput,
                                        target : _target
                                    }
                                )


        validation_DS.loadNext()

    if error_sum < error_now:
        sess.run(tf.assign(tf_validation_error, error_sum))

        # is not overtrained
        return False

    # is overtrained
    return True





with tf.Session() as sess:

    try:
        saver.restore(sess, 'saved/32hidden/model')
        print('Restoring model ...')
    except:
        sess.run(init)
        print('Initialize all variables')


    # detector = OvertrainDetector(sess, cross_entropy, model)

    for i in range(15000):

        _charInput = DS.getCharGraphInput()
        _wordInput = DS.getWordGraphInput()
        _target = DS.getTarget()

        actual_err, _ , max_args = sess.run([cross_entropy, minimize, prediction_max_arg],
                                    {
                                        word_inp: _wordInput,
                                        char_inp: _charInput,
                                        target: _target
                                    }
                                )

        TOTAL_WORDS += np.array(_target).size
        MISTAKEN_WORDS += np.not_equal(max_args, np.argmax(_target, axis=1)).astype(int).sum()

        if i % 100 == 0 and not isOvertrained(sess, cross_entropy, model, validation_DS, validation_dataset_error):
            print('Not overtrained...saving model')
            saver.save(sess, 'saved/32hidden/model')

        if i % 25 == 0:
            print('---------------------------------------------------')
            print('Prediction is = {}'.format(max_args))
            print('It should be = {}'.format(np.argmax(_target, axis=1)))
            bad_words_predictions = np.not_equal(max_args, np.argmax(_target, axis=1)).astype(int).sum()
            if bad_words_predictions:
                print(colored('{} mistakes'.format(bad_words_predictions), 'red'))
            else:
                print(colored('0 mistakes', 'green'))
            print('performance = {} %'.format(np.around(100 - MISTAKEN_WORDS*100/TOTAL_WORDS, decimals=5)))

        DS.loadNext()









#
