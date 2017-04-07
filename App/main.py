import tensorflow as tf
import numpy as np
import getTokens as tokensGetter

import char_pos_tagger as charTagger
import word_pos_tagger as wordGraph

from dataset import MODEL_CBOW, Dataset
from gensim.models import word2vec
from definedChars.charDict import CharDictionary
from OvertrainDetector import OvertrainDetector

print('--------------------------')
print('Modules loaded successfuly')

model = word2vec.Word2Vec.load_word2vec_format(MODEL_CBOW, binary=True)
charDict = CharDictionary()
DS = Dataset(model=model)

# [None, None, charDict.len]
# None - pocet slov, neviem
# None - maximalny pocet pismen -> tiez neviem aky bude najvacsi pocet pismen vo vete
# Ale viem aky bude embedding
char_inp = tf.placeholder(shape=[None, None, charDict.len], dtype=tf.float32)
char_multiplication = charTagger.charTaggingGraph(char_inp)

word_inp = tf.placeholder(shape=[1, None, 80], dtype=tf.float32)
word_multiplication = wordGraph.wordPosTagger(word_inp)

target = tf.placeholder(tf.float32, [None, 19])


final_addition = word_multiplication + char_multiplication
prediction = tf.nn.softmax(final_addition)
# shapeOfResult = tf.shape(softmax)

cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()



with tf.Session() as sess:

    try:
        saver.restore(sess, 'saved/32hidden/model')
        print('Restoring model ...')
    except:
        sess.run(init)
        print('Initialize all variables')


    detector = OvertrainDetector(sess, cross_entropy, model)

    for i in range(15000):

        _charInput = DS.getCharGraphInput()
        _wordInput = DS.getWordGraphInput()
        _target = DS.getTarget()

        actual_err, _ = sess.run([cross_entropy, minimize],
                                    {
                                        word_inp: _wordInput,
                                        char_inp: _charInput,
                                        target: _target
                                    }
                                )

        if i % 100 == 0 and not detector.overtrainDetected():
            saver.save(sess, 'saved/32hidden/model')

        if i % 25 == 0:
            print(actual_err)

        DS.loadNext()









#
