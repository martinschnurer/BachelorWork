var fs = require('fs'),
    readline = require('readline'),
    stream = require('stream'),
    logStream = fs.createWriteStream('../dataset', {'flags': 'a'});

var tokens = {
	'S' : 1,
	'A' : 2,
	'P' : 3,
	'N' : 4,
	'V' : 5,
	'D' : 6,
	'E' : 7,
	'O' : 8,
	'T' : 9,
	'J' : 10,
	'0' : 4
}

function getTag(allTags) {
    return tokens[allTags[0]] || 0
}

function sentenceBegin(line) {
    return line.indexOf('<s>') > -1
}

function sentenceEnd(line) {
    return line.indexOf('</s>') > -1
}

function getLemma(line) {
    return line.split('\t')[1]
}

function getTokens(line) {
    return line.split('\t')[2]
}

function getWord(line) {
    return line.split('\t')[0]
}


var instream = fs.createReadStream('../wiki-2014-02.ver');
var outstream = new stream;
outstream.readable = true;
outstream.writable = true;

var rl = readline.createInterface({
    input: instream,
    output: outstream,
    terminal: false
});



var sentenceReading = false
var columns
var lines = []
var words = []
var lemmas = []
var tags = []
var iteration = 0

function writeArrToFile(arr) {
    arr.forEach((word) => {
        logStream.write(word + ' ')
    })
    logStream.write('\n')
}

rl.on('line', function(line) {

    if (sentenceBegin(line)) {
        sentenceReading = true
        return
    }

    if (sentenceEnd(line)) {
        sentenceReading = false
        writeArrToFile(words)
        writeArrToFile(lemmas)
        writeArrToFile(tags)
        words = []
        lemmas = []
        tags = []
        return
    }

    columns = line.split('\t')
    if (sentenceReading) {
        words.push(columns[0])
        lemmas.push(columns[1])
        tags.push(getTag(columns[2]))
    }

});

rl.on('close', function() {
    console.timeEnd('trvanie')
})


// /*Pri pauze sa vyprazdni zasobnik a zapise sa do suboru*/
// rl.on('pause', function() {
//     lines.forEach(function(obj) {
//         obj.forEach(function(word) {
//             logStream.write(word + ' ')
//         })
//         logStream.write('\n')
//     })
//
//     lines = []
//     rl.resume()
// })
