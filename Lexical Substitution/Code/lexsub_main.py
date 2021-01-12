#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 
import string
from typing import List
from nltk.stem import WordNetLemmatizer


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.
    Replace punctuation with whitespaces and split the sentence according to whitespaces.
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = set()  # initialized as set first to avoid duplicates
    lemma.replace(' ', '_')  # replace space in the input lemma as _ to fit WordNet representation
    for lexeme in wn.lemmas(lemma, pos):
        for synset_lemma in lexeme.synset().lemmas():
            candidates.add(synset_lemma.name())
    if lemma in candidates:
        candidates.remove(lemma)
    candidates_list = list(str.replace('_', ' ') for str in candidates)  # replace _ in the WordNet representation as space
    return candidates_list


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    lemma = context.lemma
    lemma.replace(' ', '_')
    pos = context.pos

    possible_synonyms = {}

    for lexeme in wn.lemmas(lemma, pos):
        for synset_lemma in lexeme.synset().lemmas():
            count = synset_lemma.count()
            if synset_lemma.name() not in possible_synonyms:
                possible_synonyms[synset_lemma.name()] = count
            else:
                possible_synonyms[synset_lemma.name()] += count

    if lemma in possible_synonyms:
        del possible_synonyms[lemma]

    res = max(possible_synonyms, key=possible_synonyms.get).replace('_', ' ')
    return res
# Result for Part 2
# Total = 298, attempted = 298
# precision = 0.098, recall = 0.098
# Total with mode 206 attempted 206
# precision = 0.136, recall = 0.136


def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    lemma = context.lemma
    pos = context.pos
    left_context = context.left_context
    right_context = context.right_context

    context_words = tokenize(' '.join(left_context + right_context))
    context_words = [word for word in context_words if word not in stop_words]
    context_words = [word for word in context_words if word.isalpha()]
    # context_words = [wnl.lemmatize(word) for word in context_words]  # lemmatize using NLTK funciton
    context_words = set(context_words)

    overlaps = []

    for lexeme in wn.lemmas(lemma, pos):
        synset = lexeme.synset()
        # Extensions: definition and all examples for the synset
        def_exp = list()
        def_exp.append(synset.definition())
        def_exp += synset.examples()
        # Extensions: definition and all examples for all hypernyms of the synset
        hypernyms = synset.hypernyms()
        if hypernyms:
            for i in range(len(hypernyms)):
                def_exp.append(hypernyms[i].definition())
                def_exp += hypernyms[i].examples()

        def_exp = ' '.join(def_exp)
        def_exp_words = tokenize(def_exp)
        def_exp_words = [word for word in def_exp_words if word not in stop_words]
        def_exp_words = [word for word in def_exp_words if word.isalpha()]
        def_exp_words = [wnl.lemmatize(word) for word in def_exp_words]  # lemmatize using NLTK funciton
        def_exp_words = set(def_exp_words)

        overlap = def_exp_words.intersection(context_words)
        if len(overlap) != 0:
            overlaps.append(synset)

    # If overlaps existed, find the most frequent lemma_name which != target_word (lemma)
    if overlaps:
        lexeme_freq = {}
        for synset in overlaps:
            for syn_lemma in synset.lemmas():
                lemma_name = syn_lemma.name()
                if lemma_name.lower() != lemma:
                    lemma_name = lemma_name.replace('_', ' ')
                    lexeme_freq[lemma_name] = lexeme_freq.get(lemma_name, 0) + 1
        # If such lemma_name existed, sort, and choose the most frequent one
        if len(lexeme_freq) != 0:
            lexeme_freq_list = [(lemma_name, freq) for lemma_name, freq in lexeme_freq.items()]
            lexeme_freq_list.sort(key=lambda x: x[1], reverse=True)
            return lexeme_freq_list[0][0]
        else:
            return wn_frequency_predictor(context)
    else:
        return wn_frequency_predictor(context)
# Result for Part 3
# Total = 298, attempted = 298
# precision = 0.090, recall = 0.090
# Total with mode 206 attempted 206
# precision = 0.131, recall = 0.131


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context : Context) -> str:
        # Part 4
        possible_synonyms = [s.replace(' ', '_') for s in get_candidates(context.lemma, context.pos)]
        nearest = max(possible_synonyms, key=lambda x: self.model.similarity(context.lemma, x) if x in self.model.vocab else 0)
        if nearest:
            return nearest
        else:
            return None
# Result for Part 4
# Total = 298, attempted = 298
# precision = 0.115, recall = 0.115
# Total with mode 206 attempted 206
# precision = 0.170, recall = 0.170


    def my_improved_predictor(self, context : Context) -> str:
        # Part 6
        '''
        Main idea for the improved predictor:
        Set a context window size to 4 and combine with the original word to form as a sentence. Generate the sentence vector using
        Word2Vec. Compare the similarity between candidate synonyms and the sentence. Return the one with the highest similarity.
        '''
        stop_words = set(stopwords.words('english'))
        lemma = context.lemma
        pos = context.pos

        possible_synonyms = get_candidates(lemma, pos)
        left = [w for w in context.left_context if w not in string.punctuation]
        right = [w for w in context.right_context if w not in string.punctuation]
        sentence = left[-4:] + [context.word_form] + right[0:4]
        sentence = [w for w in sentence if w not in stop_words or w == context.word_form]
        sentence_vec = np.zeros(300)

        w_index = sentence.index(context.word_form)
        count = 0
        for w in sentence:
            try:
                sentence_vec = sentence_vec + np.exp(-abs(count - w_index) ** 2) * self.model.wv[w]
            except:
                pass
            count += 1

        cos_similarity = {}
        for synonym in possible_synonyms:
            try:
                synonym_vec = self.model.wv[synonym]
                cos_similarity[synonym] = np.dot(synonym_vec, sentence_vec) / (np.linalg.norm(synonym_vec) * np.linalg.norm(sentence_vec))
            except:
                continue
        if not cos_similarity:
            res = possible_synonyms[0]
        else:
            res = max(cos_similarity, key=cos_similarity.get)
        return res
# Result for Part 6
# Total = 298, attempted = 298
# precision = 0.136, recall = 0.136
# Total with mode 206 attempted 206
# precision = 0.209, recall = 0.209


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Part 5
        left_context = context.left_context
        right_context = context.right_context

        # left_context_str = ' '.join(left_context)
        # mask_index = len(self.tokenizer.encode(left_context_str)) - 1

        input = ' '.join(left_context + ['[MASK]'] + right_context)
        input_toks = self.tokenizer.encode(input)
        input_convert = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_index = input_convert.index('[MASK]')

        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_index])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)

        possible_synonyms = [s.replace(' ', '_') for s in get_candidates(context.lemma, context.pos)]
        possible_synonyms_sorted = []
        for word in best_words:
            if word in possible_synonyms:
                possible_synonyms_sorted.append(word)

        res = possible_synonyms_sorted[0]
        return res
# Result for Part 5
# Total = 298, attempted = 298
# precision = 0.115, recall = 0.115
# Total with mode 206 attempted 206
# precision = 0.170, recall = 0.170


    

if __name__=="__main__":

    # This program should run on my best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # prediction = smurf_predictor(context)
        # prediction = wn_frequency_predictor(context)  # Part 2
        # prediction= wn_simple_lesk_predictor(context)  # Part 3
        # prediction = predictor.predict_nearest(context)  # Part 4

        # bertpredictor = BertPredictor()  # Part 5
        # prediction = bertpredictor.predict(context)  # Part 5

        prediction = predictor.my_improved_predictor(context)   # Part 6
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
