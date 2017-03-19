from random import shuffle

SENTENCES_COUNT = 4039850
BATCH_FROM_FILE = 100000

dataset_path = '../wiki-2017-02.ver'
fd = open(dataset_path, 'r')


def getTokens():
    tokens = {}
    with open('../../tokeny_vyznam') as f:
        for line in f:
            arr = line.split('-')
            # print(arr)
            arr[-1] = arr[-1].strip()
            tokens[arr[0]] = { 'name':'', 'number' : -1}
            tokens[arr[0]]['name'] = arr[1]
            tokens[arr[0]]['number'] = int(arr[2])
    return tokens



"""
This loads from file N sentences and return as array

returns [
            [
                [Word1,Word2,Word3],
                [1,5,8]
            ],
            [
                [Word4,Word5,Word6,Word7],
                [2,1,5,8]
            ],
        ]
"""
def loadSentences(f, tokens, n):
    sentences = []
    while len(sentences) < n:
        line = f.readline()
        if line[0] == '<' and '<s>' in line:
            sentence = [[],[]]
            while True:
                line = f.readline()
                if '</s>' in line:
                    break
                word,_,info,_ = line.split('\t')
                sentence[0].append(word)
                sentence[1].append(tokens[info[0]]['number'])
            sentences.append(sentence)
    return sentences


# Given sentences split into more files for train_dataset, validation dataset.
# and if given testing_dataset, then into test_dataset total_words
#
# If shuffle parameter is True, before saving to files, all sentences will be
# shuffled
def splitSentences(sentences, train_fd, validation_fd, test_fd=None, shuffleDataset=False):
    if shuffleDataset:
        shuffle(sentences)

    whole_len = len(sentences)
    train_len = round(whole_len * 0.8)
    validation_len = round(whole_len * 0.1)
    test_len = whole_len - (train_len + validation_len)

    assert train_len + validation_len + test_len == whole_len

    train_sentences = sentences[:train_len]
    validation_sentences = sentences[train_len:train_len + validation_len]
    test_sentences = sentences[train_len + validation_len:]

    assert len(train_sentences) + len(validation_sentences) + len(test_sentences) == len(sentences)

    for s in train_sentences:
        for w in s[0]:
            train_fd.write("{} ".format(w))
        train_fd.write('\n')
        for wt in s[1]:
            train_fd.write("{} ".format(wt))
        train_fd.write('\n')

    for s in validation_sentences:
        for w in s[0]:
            validation_fd.write("{} ".format(w))
        validation_fd.write('\n')
        for wt in s[1]:
            validation_fd.write("{} ".format(wt))
        validation_fd.write('\n')

    for s in test_sentences:
        for w in s[0]:
            test_fd.write("{} ".format(w))
        test_fd.write('\n')
        for wt in s[1]:
            test_fd.write("{} ".format(wt))
        test_fd.write('\n')

    return True


###############################################################################
###############################################################################
###############################################################################

tokens = getTokens()

first = open('test/1','w')
second = open('test/2','w')
third = open('test/3','w')

remaining = SENTENCES_COUNT
all_batches = int(SENTENCES_COUNT / BATCH_FROM_FILE)

for batch in range(all_batches):
    print("{} / {}".format(batch, all_batches))
    loaded_sentences = loadSentences(fd, tokens, min(BATCH_FROM_FILE, remaining))
    splitSentences( loaded_sentences,
                    first,
                    second,
                    third,
                    shuffleDataset=True
                    )
    remaining = remaining - BATCH_FROM_FILE
