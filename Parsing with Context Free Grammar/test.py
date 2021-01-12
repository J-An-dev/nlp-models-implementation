from grammar import Pcfg
from math import fsum

"""
Part 1
"""
with open('atis3.pcfg','r') as grammar_file:
    grammar = Pcfg(grammar_file)

print(grammar.rhs_to_rules[('flight',)])




