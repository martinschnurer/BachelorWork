import os
from gensim.models import word2vec
from getTokens import getTokens
from char_pos_tagger import getVectorFromSentence

MODEL_CBOW = os.path.join(os.path.dirname(__file__), 'model/80cbow.bin')
TRAIN_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/1')
VALIDATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/2')
TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/3')


TOKENS = getTokens()
WORD_VEC_SIZE = 80


class Dataset:
    def __init__(self, file_dataset=None, model = None):
        # dataset file and word2vec model
        self.file_dataset = file_dataset if file_dataset else open(TRAIN_DATASET_PATH, 'r')
        self.model = model

        self.sentence = None
        self.target = None
        self.vec_for_char_graph = None
        self.vec_for_word_graph = None
        self.vec_target = None

        self.loadNext()

    def _oneHot(self, which, size=19):
        which = which - 1
        vec = [0] * size
        vec[which] = 1
        return vec


    def _getZerosVec(self, size):
        return [0] * size


    def _getWordEmbedding(self, word):
        if word in self.model:
            return self.model[word]
        else:
            return self._getZerosVec(WORD_VEC_SIZE)


    def getCharGraphInput(self):
        return getVectorFromSentence(self.sentence)


    def getWordGraphInput(self):
        if self.model == None:
            return None

        out_vector = []
        for w in self.sentence:
            out_vector.append(self._getWordEmbedding(w))
        return [out_vector]


    def getTarget(self):
        return list(map(self._oneHot, self.target))

    def loadNext(self):
        self.sentence = self.file_dataset.readline().rstrip().rstrip().split('\t')
        self.target = self.file_dataset.readline().rstrip().rstrip().split()
        self.target = list(map(int, self.target))
        return self


if __name__ == "__main__":

    dataset_file = open(TRAIN_DATASET_PATH, 'r')
    # model = word2vec.Word2Vec.load_word2vec_format(MODEL_CBOW, binary=True)

    ds = Dataset(dataset_file)

    print(ds.getWordGraphInput())
    print(ds.getCharGraphInput())
    print(ds.getTarget())
