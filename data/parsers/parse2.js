/**
 * Toto je parser pre dataset
 * ktory bude pouzivany pre
 * NN na urcovanie slovneho druhu pre
 * jedno nezavisle slovo
 */

 var fs = require('fs'),
     readline = require('readline'),
     stream = require('stream'),
     logStream = fs.createWriteStream('../dataset2', {'flags': 'a', 'defaultEncoding': 'utf8'});

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

// vsetky rozne pismena a,b,c,d,e,f,g,h,...ž,š,č,
var allChars = new Set()


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


function writeWordToFile(word, tag, length, arr) {
	logStream.write(word+'\n')
    logStream.write(tag+'\n')
	logStream.write(length+'\n')
	arr.forEach(function(element) {
		logStream.write(element + ' ')
	})
	logStream.write('\n')
}


console.time('trvanie')
let iteration = 0

let word,
	i,
	len,
	tmp_arr,
    tag;


rl.on('line', function(line) {
	arr = []

	if (line.indexOf('<') > -1 || line.indexOf('>') > -1) {
		return
	}

	// if (iteration++ > 100000) {
	// 	rl.close()
	// }

	// Ziskaj slovo z vety
	word = getWord(line)
    tag = getTag(getTokens(line))

	// Zisti dlzku slova
	len = word.length

	// ROZLOZ STRING NA POLE PISMEN
	tmp_arr = word.split('')

	// PRIDAJ DO SETU kazde pismeno
	tmp_arr.forEach(function (char) {
		allChars.add(char)
	})

	writeWordToFile(word, tag, len, tmp_arr)
});


rl.on('close', function() {
	console.timeEnd('trvanie')

	sorted_arr = [...allChars].sort(function(a,b) {
		return a > b ? 1 : -1
	})
	console.log(sorted_arr)

})
