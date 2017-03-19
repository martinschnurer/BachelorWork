#!/usr/bin/python3
import struct
from gensim.models import word2vec


def findSpace(binaryLine):
	space = ord(' ')
	counter = 0
	for c in binaryLine:
		if c == space:
			return counter
		counter = counter + 1
	return -1

model = word2vec.Word2Vec.load_word2vec_format('80cbow.bin', binary=True)
print("Model loaded")
