# Analýza spracovania jazyka

# 1.Representation

## 1.1 Typy využitia
- __Automatic summarization__ ->   Produce a readable summary of a chunk of text. Often used to provide summaries of text of a known type, such as articles in the financial section of a newspaper.

- __Coreference resolution__ ->
    Given a sentence or larger chunk of text, determine which words ("mentions") refer to the same objects ("entities"). Anaphora resolution is a specific example of this task, and is specifically concerned with matching up pronouns with the nouns or names that they refer to. The more general task of coreference resolution also includes identifying so-called "bridging
     relationships" involving referring expressions. For example, in a sentence such as "He entered John's house through the front door", "the front door" is a referring expression and the bridging relationship to be identified is the fact that the door being referred to is the front door of John's house (rather than of some other structure that might also be referred to).

- __Discourse analysis__ ->
    This rubric includes a number of related tasks. One task is identifying the discourse structure of connected text, i.e. the nature of the discourse relationships between sentences (e.g. elaboration, explanation, contrast). Another possible task is recognizing and classifying the speech acts in a chunk of text (e.g. yes-no question, content question, statement, assertion, etc.).
- __Question answering__ ->
    Given a human-language question, determine its answer. Typical questions have a specific right answer (such as "What is the capital of Canada?"), but sometimes open-ended questions are also considered (such as "What is the meaning of life?"). Recent works have looked at even more complex questions.

- __Relationship extraction__ ->
    Given a chunk of text, identify the relationships among named entities (e.g. who is married to whom).

- __Sentence breaking__ ->
    Given a chunk of text, find the sentence boundaries. Sentence boundaries are often marked by periods or other punctuation marks, but these same characters can serve other purposes (e.g. marking abbreviations).

- __keyborad typping__ -> wrong key pressed prediction, auto-correction of word, learning from user typing

- __OWN__ -> Chat conversation prediction, predikcia dalsich slov resp. viet, resp. temy na zaklade poslednych viet.


## 1.2 Word2Vec

- __1.2.1 Quora:__

word2vec is not one algorithm.

word2vec is not deep learning.

The recommended algorithm in word2vec, skip-gram with negative sampling, factorizes a word-context PMI matrix.

Another thing that's important to understand, is that word2vec is not deep learning; both CBOW and skip-gram are "shallow" neural models. Tomas Mikolov even told me that the whole idea behind word2vec was to demonstrate that you can get better word representations if you trade the model's complexity for efficiency, i.e. the ability to learn from much bigger datasets.



## 2. Plán

- [ ] Prehľadať väčšinu možností využitia
- [ ] Pochopit reprezentaciu slov/viet do vektora


extraktivna  -> takze toto robit
abstraktivna


identifikovat klucove slova...
toto robili aj minulorocny => +oprava gramatickych chyb,
## urcovanie slovnych druhov


rekurentna neuronova siet na premenlivu dlzku textu
prvych N slov z dokumentu
====>  [text.fiit.stuba.sk]

POZRIET SI VSETKO O URCOVANI SLOVNYCH DRUHOCH
POUZIT REKURENTNU NEURONOVU SIET
SMERODAJNE CRTY SLOVA KTORE POMOZU URCIT SLOVNY DRUH


-> ROZBEHAT TENSORFLOW


CPU ONLY NIE GPU
inicializacia premennych, priradenie konstant pre tf.

online localhost interpreter
__jupyter notebook__



Recurrent Neural Networks (RNN)

Convolutional Neural Networks—and more generally, feed-forward neural networks—do not traditionally have a notion of time or experience unless you explicitly pass samples from the past as input. After they are trained, given an input, they treat it no differently when shown the input the first time or the 100th time. But to tackle some problems, you need to look at past experiences and give a different answer.



Long Short Term Memory (LSTM)

RNNs keep context in their hidden state (which can be seen as memory). However, classical recurrent networks forget context very fast. They take into account very few words from the past while doing prediction. Here is an example of a language modelling problem that requires longer-term memory.

    I bought an apple … I am eating the _____

The probability of the word “apple” should be much higher than any other edible like “banana” or “spaghetti”, because the previous sentence mentioned that you bought an “apple”. Furthermore, any edible is a much better fit than non-edibles like “car”, or “cat”.

Long Short Term Memory (LSTM) units try to address the problem of such long-term dependencies. LSTM has multiple gates that act as a differentiable RAM memory. Access to memory cells is guarded by “read”, “write” and “erase” gates. Information stored in memory cells is available to the LSTM for a much longer time than in a classical RNN, which allows the model to make more context-aware predictions. An LSTM unit is shown in Figure 5.

__http://colah.github.io/posts/2015-08-Understanding-LSTMs/__
[-http://colah.github.io/posts/2015-08-Understanding-LSTMs/]



---------------------------------------------------------------------------------------------------------
NGRAMS
---------------------------------------------------------------------------------------------------------


 What are N-Grams?
N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). For example, for the sentence "The cow jumps over the moon". If N=2 (known as bigrams), then the ngrams would be:

    the cow
    cow jumps
    jumps over
    over the
    the moon

So you have 5 n-grams in this case. Notice that we moved from the->cow to cow->jumps to jumps->over, etc, essentially moving one word forward to generate the next bigram.

If N=3, the n-grams would be:

    the cow jumps
    cow jumps over
    jumps over the
    over the moon

So you have 4 n-grams in this case. When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.

How many N-grams in a sentence?
If X=Num of words in a given sentence K, the number of n-grams for sentence K would be:


What are N-grams used for?
N-grams are used for a variety of different task. For example, when developing a language model, n-grams are used to develop not just unigram models but also bigram and trigram models. Google and Microsoft have developed web scale n-gram models that can be used in a variety of tasks such as spelling correction, word breaking and text summarization. Here is a publicly available web scale n-gram model by Microsoft: http://research.microsoft.com/en-us/collaboration/focus/cs/web-ngram.aspx. Here is a paper that uses Web N-gram models for text summarization:Micropinion Generation: An Unsupervised Approach to Generating Ultra-Concise Summaries of Opinions

Another use of n-grams is for developing features for supervised Machine Learning models such as SVMs, MaxEnt models, Naive Bayes, etc. The idea is to use tokens such as bigrams in the feature space instead of just unigrams. But please be warned that from my personal experience and various research papers that I have reviewed, the use of bigrams and trigrams in your feature space may not necessarily yield any significant improvement. The only way to know this is to try it!

Java Code Block for N-gram Generation
This code block generates n-grams at a sentence level. The input consists of N (the size of n-gram), sent the sentence and ngramList a place to store the n-grams generated.


private static void generateNgrams(int N, String sent, List ngramList) {
  String[] tokens = sent.split("\\s+"); //split sentence into tokens

  //GENERATE THE N-GRAMS
  for(int k=0; k<(tokens.length-N+1); k++){
    String s="";
    int start=k;
    int end=k+N;
    for(int j=start; j<end; j++){
     s=s+""+tokens[j];
    }
    //Add n-gram to a list
    ngramList.add(s);
  }
}//End of method



-------------------------------------------------------------------------------



The tf.train.AdamOptimizer uses Kingma and Ba's Adam algorithm to control the learning rate. Adam offers several advantages over the simple tf.train.GradientDescentOptimizer. Foremost is that it uses moving averages of the parameters (momentum); Bengio discusses the reasons for why this is beneficial in Section 3.1.1 of this paper. Simply put, this enables Adam to use a larger effective step size, and the algorithm will converge to this step size without fine tuning.

The main down side of the algorithm is that Adam requires more computation to be performed for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient); and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter). A simple tf.train.GradientDescentOptimizer could equally be used in your MLP, but would require more hyperparameter tuning before it would converge as quickly.


--------------------------------------------------------------------------------
