import numpy as np

def getCharDict(path):
    i = 1
    dictionary = {}
    file = open('definedChars/chars') if not path else open(path)
    line = file.readline()[:-1] # get all chars, except newline char
    for char in line:
        dictionary[char] = i
        i = i + 1
    return dictionary



class CharDictionary:
    def __init__(self, path):
        self.dictionary = getCharDict(path)
        self.len = len(self.dictionary) + 1 # additional position for undefined char

    def __contains__(self, char):
        return char in self.dictionary

    def getEmbedding(self, char):
        if char in self.dictionary:
            return self.dictionary[char]
        else:
            return 0

    def getMaxLength(self, sentence):
        max_length = 0
        for word in sentence:
            size = len(word)
            if size > max_length:
                max_length = size
        return max_length

    def getVectorFromSentence(self, sentence, numpy=True):
        sentence_arr = []
        max_length = self.getMaxLength(sentence)
        for word in sentence:
            sentence_arr.append(self.getVectorFromWord(word, size=max_length))
        return np.array(sentence_arr) if numpy else sentencec_arr

    def getVectorFromWord(self, word, size, numpy=True):
        word_arr = []
        for char in word:
            word_arr.append(self.getVectorFromChar(char))
        for _ in range(size - len(word_arr)):
            word_arr.append([0] * self.len)
        return np.array(word_arr) if numpy else word_arr

    def getVectorFromChar(self, char, numpy=True):
        vec = [0] * self.len
        charPosition = self.getEmbedding(char)
        vec[charPosition] = 1
        return np.array(vec) if numpy else vec

    def getVector(self, char):
        vec = [0] * self.len
        charPosition = self.getEmbedding(char)
        vec[charPosition] = 1
        return vec

if __name__ == '__main__':
    charDict = CharDictionary('./chars')
