import tensorflow as tf
import numpy as np
from definedChars.charDict import CharDictionary
import getTokens as tokensGetter
from gensim.models import word2vec

import char_pos_tagger as charTagger
import word_pos_tagger as wordGraph
from dataset import MODEL_CBOW, Dataset

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


addition = word_multiplication + char_multiplication
softmax = tf.nn.softmax(addition)
shapeOfResult = tf.shape(softmax)

prediction = tf.nn.softmax(multiplication)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

optimizer = tf.train.AdamOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    _charInput = DS.getCharGraphInput()
    _wordInput = DS.getWordGraphInput()
    _target = DS.getTarget()

    vysledok, sejp = sess.run([softmax, shapeOfResult],
            {
                word_inp: _wordInput,
                char_inp: _charInput,
                target: _target
            }
        )

    print(sejp)
    print(vysledok)







#
