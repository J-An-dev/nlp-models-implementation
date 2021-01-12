from lexsub_main import get_candidates
from collections import defaultdict
from nltk.corpus import wordnet as wn
from lexsub_main import tokenize

# Part 1
# list = get_candidates('turn','v')
# print(list)

# Part 2
# def wn_frequency_predictor(lemma, pos) -> str:
#
#     # lemma = context.lemma
#     lemma.replace(' ', '_')
#     # pos = context.pos
#     possible_synonyms = {}
#
#     for lexeme in wn.lemmas(lemma, pos):
#         for synset_lemma in lexeme.synset().lemmas():
#             count = synset_lemma.count()
#             # if synset_lemma.name() == lemma:
#             #     continue
#             if synset_lemma.name() in possible_synonyms:
#                 possible_synonyms[synset_lemma.name()] += count
#             else:
#                 possible_synonyms[synset_lemma.name()] = count
#
#     if lemma in possible_synonyms:
#         del possible_synonyms[lemma]
#
#     res = max(possible_synonyms, key=possible_synonyms.get).replace('_', ' ')
#     return res
#
# possible_synonym = wn_frequency_predictor('turn', 'v')
# print(possible_synonym)

# Part 3
# example = "Today is a good day, and we'd like to go shopping. But, after a sudden, the rain pouring dowm!"
# print(tokenize(example))

# Tokenize
s = "i like to go Mr.An, but it's hard for me!"
print(tokenize(s))