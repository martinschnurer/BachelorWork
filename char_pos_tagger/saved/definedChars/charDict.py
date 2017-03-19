
def getCharDict():
    file = open('./chars')
    line = file.readline()[:-1]
    print(line)

if name == '__name__':
    getCharDict()
