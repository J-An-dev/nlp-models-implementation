import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of data structures in Part 3 ###
def check_table_format(table):
    """
    Return True if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((A, i, k),(B, k, j)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((A, i, k),(B, k, j)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((A, i, k),(B, k, j)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((A, i, k),(B, k, j)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing no-terminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # Part 2
        table, probs = self.parse_with_backpointers(tokens)
        if self.grammar.startsymbol in table[(0, len(tokens))]:
            return True
        else:
            return False

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # Part 3
        table = defaultdict(dict)
        probs = defaultdict(dict)
        n = len(tokens)

        # Initialization:
        for i in range(0, n):
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
            # e.g. rule = ('FLIGHT', ('flight',), 1.0)
                table[(i,i+1)][rule[0]] = tokens[i]
                probs[(i,i+1)][rule[0]] = math.log(rule[2])

        # Main update loop:
        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                for k in range(i+1, j):
                    for row in table[(i,k)]:
                        for col in table[(k,j)]:
                            for rule in self.grammar.rhs_to_rules[(row,col)]:
                                prob = math.log(rule[2]) + probs[(i,k)][row] + probs[(k,j)][col]
                                if (rule[0] not in table[(i,j)]) or \
                                   ((probs[(i,j)][rule[0]] != 0) and (probs[(i,j)][rule[0]] < prob)):
                                    table[(i, j)][rule[0]] = ((row, i, k), (col, k, j))
                                    probs[(i, j)][rule[0]] = prob

        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # Part 4
    bps = chart[(i,j)][nt]
    if (j-i) == 1:  # Diagonal Terminal (Leaf nodes may be strings)
        return (nt, bps)
    else:
        l_tree = bps[0]
        r_tree = bps[1]
        return (nt, get_tree(chart, l_tree[1], l_tree[2], l_tree[0]), get_tree(chart, r_tree[1], r_tree[2], r_tree[0]))
 
       
if __name__ == "__main__":
    
    # with open('atis3.pcfg','r') as grammar_file:
    #     grammar = Pcfg(grammar_file)
    #     parser = CkyParser(grammar)
    #     toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
    #     # toks = ['miami', 'flights','cleveland', 'from', 'to','.']
    #     table, probs = parser.parse_with_backpointers(toks)
    #
    #     print(parser.is_in_language(toks))
    #     print(check_table_format(table))
    #     print(check_probs_format(probs))
    #
    #     print(get_tree(table, 0, len(toks), grammar.startsymbol))

    with open('atis_play.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks = ['she', 'saw', 'the', 'cat', 'with', 'glasses']
        table, probs = parser.parse_with_backpointers(toks)

        print(parser.is_in_language(toks))
        print(check_table_format(table))
        print(check_probs_format(probs))

        print(table[(0,1)])
        print(table[(0,4)])
        print(table[(1,6)])

        print(probs[(0,1)])
        print(probs[(0,4)])
        print(probs[(1,6)])

        print(get_tree(table, 0, len(toks), grammar.startsymbol))

        
