"""""
Part 1
"""""
# from trigram_model import get_ngrams
# sequence = ["natural","language","processing"]
# result = get_ngrams(sequence, 2)
# print(result)


"""""
Part 2
"""""
# from trigram_model import corpus_reader
# generator = corpus_reader('./data/brown_test.txt')
# for sentence in generator:
#     print(sentence)

# from trigram_model import TrigramModel
# model = TrigramModel('./data/brown_train.txt')
# print(model.trigramcounts[('START','START','the')])
# print(model.bigramcounts[('START','the')])
# print(model.unigramcounts[('START',)])
# print(model.n_sentences)
# print("\n")
# print(sum(model.unigramcounts.values()))
# print(model.n_words)


"""""
Part 3
"""""
# from trigram_model import TrigramModel
# model = TrigramModel('./data/brown_train.txt')
# print(model.raw_unigram_probability(('the',)))
# print(model.raw_bigram_probability(('START','the')))
# print(model.raw_trigram_probability(('START','START','the')))

"""""
Part 4
"""""
# from trigram_model import TrigramModel
# model = TrigramModel('./data/brown_train.txt')
# print(model.smoothed_trigram_probability(('lazy','dog','STOP')))

"""""
Part 5
"""""
# from trigram_model import TrigramModel
# model = TrigramModel('./data/brown_train.txt')
# print(model.sentence_logprob(['the','quick','brown','fox','jumps','over','the','lazy','dog']))

"""""
Optional
"""""
# from trigram_model import TrigramModel
# model = TrigramModel('./data/brown_train.txt')
# new_sentence = model.generate_sentence()
# print(new_sentence)


