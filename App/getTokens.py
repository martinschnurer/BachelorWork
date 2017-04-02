import os

TOKENY_VYZNAM_PATH = os.path.join(os.path.dirname(__file__), './tokeny_vyznam.csv')

def getTokens():
    tokens = {}
    with open(TOKENY_VYZNAM_PATH) as f:
        for line in f:
            arr = line.split(',')
            arr[-1] = arr[-1].strip()
            tokens[arr[0]] = { 'name':'', 'number' : -1}
            tokens[arr[0]]['name'] = arr[1]
            tokens[arr[0]]['number'] = int(arr[2])
    return tokens
