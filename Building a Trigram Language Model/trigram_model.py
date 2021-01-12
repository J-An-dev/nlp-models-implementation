import sys
from collections import defaultdict
import math
import random
import numpy as np
import os
import os.path


def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    PART 1
    Given a sequence, this function returns a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    START = ['START'] + ['START' for i in range(n-2) if n > 2]
    STOP = ['STOP']
    seq = START + sequence + STOP
    res = []
    for i in range(len(seq) + 1 - n):
        res.append(tuple(seq[i:i+n]))

    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count n-grams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        PART 2
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            for unigram in get_ngrams(sentence, n=1):
                self.unigramcounts[unigram] += 1
            for bigram in get_ngrams(sentence, n=2):
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, n=3):
                self.trigramcounts[trigram] += 1

        # Compute the total number of sentences
        self.n_sentences = self.unigramcounts[("START",)]

        # Compute the total number of words （include STOP exclude START）
        self.n_words = sum(self.unigramcounts.values()) - self.n_sentences 
        

    def raw_trigram_probability(self, trigram):
        """
        PART 3
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[:2] == ('START', 'START'):
            return self.trigramcounts[trigram] / self.n_sentences
        elif self.bigramcounts[trigram[:2]] != 0:
            return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]
        else:
            return 0.0

    def raw_bigram_probability(self, bigram):
        """
        PART 3
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram[:1] == ('START',):
            return self.bigramcounts[bigram] / self.n_sentences
        elif self.unigramcounts[bigram[:1]] != 0:
            return self.bigramcounts[bigram] / self.unigramcounts[bigram[:1]]
        else:
            return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        PART 3
        Returns the raw (unsmoothed) unigram probability.
        """

        # Compute the total number of words once, store in the TrigramModel instance, and then re-use it.
        if unigram == ('START',) or unigram == ('STOP',):
            return self.n_sentences / self.n_words
        else:
            return self.unigramcounts[unigram] / self.n_words

    def generate_sentence(self,t=20): 
        """
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html
        # numpy.random.choice function document

        # exclude 'UNK' and punctuation from being generated
        generated_trigram = (None, 'START', 'START')
        res = []
        i = 0
        while generated_trigram[2] != 'STOP' and i < t:
            candidate_list = [trigram for trigram in self.trigramcounts.keys() if trigram[:2] == (generated_trigram[1], generated_trigram[2])]
            prob_list = [self.raw_trigram_probability(trigram) for trigram in candidate_list]
            r = np.random.choice([candidate[2] for candidate in candidate_list], 1, p=prob_list)[0]

            if r in ['UNK', '.', ',', '(', ')', '``', '"', '!', '?', "''", ';', '.264']:
                if prob_list == [1.0]:
                    # key point here: if there is only one 'UNK' or punctuation symbol following the last two symbols in generated_trigram
                    # in the training set, then rollback one step and delete the last item in res (in some edge cases, rollback one step is not even enough)
                    # .264 is a special edge case, to simplify, just exclude it here (search .264 in brown_train.txt for more info)
                    generated_trigram = (generated_trigram[0], generated_trigram[0], generated_trigram[1])
                    res.pop()
                    i -= 1
            else:
                generated_trigram = (generated_trigram[1], generated_trigram[2], r)
                res.append(r)
                i += 1
                print(i)

        return res

    def smoothed_trigram_probability(self, trigram):
        """
        PART 4
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        res = lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[1:]) + lambda3 * self.raw_unigram_probability(trigram[-1:])
        return res
        
    def sentence_logprob(self, sentence):
        """
        PART 5
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, n=3)
        probs = [self.smoothed_trigram_probability(trigram) for trigram in trigrams]
        return float(sum(math.log2(prob) for prob in probs if prob>0))

    def perplexity(self, corpus):
        """
        PART 6
        Returns the log probability of the corpus.
        """
        l = 0
        m = 0
        M = 0
        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            m += 1
            M += len(sentence)
        l = l / (M+m)   # include the number of STOP in the final total number of word tokens
        return float(pow(2, -l))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):  #high level
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += (pp1 < pp2)

    
        for f in os.listdir(testdir2):  #low level
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            correct += (pp2 < pp1)
        
        return correct / total

if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1])

    model = TrigramModel('./data/brown_train.txt')

    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive Python prompt. 

    
    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)

    dev_corpus = corpus_reader('./data/brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment:
    dic = './data/ets_toefl_data/'
    acc = essay_scoring_experiment(dic+'train_high.txt', dic+'train_low.txt', dic+'test_high', dic+'test_low')
    print(acc)

