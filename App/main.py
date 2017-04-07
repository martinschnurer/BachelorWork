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


# np.random.random_integers(0,5,[5,3])
dataset_words_tagger = np.random.random_integers(-5, 5, [5, 5, 80])

sentences = [
'Martin nebol včera večer doma',
'Dnes je krásny slnečný deň',
'Dušan Ravolský vie krásne spievať',
'Mirinda mi neprišla vôbec vhod',
'Kikirikíííííí zakotkodákal kohút hneď ráno'
]
sentences = [ s.split() for s in sentences ]

char_inp = tf.placeholder(shape=[None, None, charDict.len], dtype=tf.float32)
char_multiplication = charTagger.charTaggingGraph(char_inp)


word_inp = tf.placeholder(shape=[1, None, 80], dtype=tf.float32)
# word_data = np.random.random_integers(-5, 5, [9, None, 80])
word_multiplication = wordGraph.wordPosTagger(word_inp)


addition = word_multiplication + char_multiplication
softmax = tf.nn.softmax(addition)

shapeOfResult = tf.shape(softmax)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _charInput = DS.getCharGraphInput()
    _wordInput = DS.getWordGraphInput()


    vysledok, sejp = sess.run([softmax, shapeOfResult],
            {
                word_inp: _wordInput,
                char_inp: _charInput
            }
        )

    print(sejp)
    print(vysledok)







#
