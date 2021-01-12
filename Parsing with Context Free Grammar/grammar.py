import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)


    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # Part 1

        # Check 1: all non-terminal symbols in lhs are upper-case
        # Check 2: rhs is
        #           - a tuple of 2 non-terminals symbols ('', '') in upper-case
        #           - or a tuple of 1 terminal symbol ('',) in lower-case
        #           - special case: 'WHNP', ('0',) & 'PUN', ('.',)
        # Check 3: all probabilities for the same lhs symbol sum to 1.0

        # Initialization
        res = True

        for i in self.lhs_to_rules.items():
            lhs = i[0]
            rules = i[1]

            # Check 1
            if not lhs.isupper():
                print("lhs non-terminal symbol (" + lhs + ") is not all upper-case")
                res = False

            # Check 2
            for rule in rules:
                rhs = rule[1]
                if (len(rhs) != 1) and (len(rhs) != 2):
                    print("rhs " + str(rhs) + " have more than 2 subtrees in rule " + str(rule))
                    res = False
                if (len(rhs) == 1) and (rhs[0] not in ['0','.']) and (not rhs[0].islower()):
                    print("rhs terminal symbol (" + rhs[0] + ") is not all lower-case in rule " + str(rule))
                    res = False
                if (len(rhs) == 2) and ((not rhs[0].isupper()) or (not rhs[1].isupper())):
                    print("rhs non-terminal symbols (" + rhs[0] + ", " + rhs[1] + ") are not all upper-case in rule " + str(rule))
                    res = False

            # Check 3
            for i in self.lhs_to_rules.items():
                lhs = i[0]
                rules = i[1]
                prob_list = [rule[2] for rule in rules]
                prob_sum = fsum(prob_list)
                if (abs(prob_sum - 1.0) > 1e-10):
                    print("all probabilities for the same lhs symbol (" + lhs + ") is not sum to 1.0")
                    res = False

            return res
                # tolerance threshold set up based on:
                # ADJP:
                # 1.0000000000012
                # ADVP:
                # 1.0000000000001
                # FRAG:
                # 0.9999999999996
                # FRAGBAR:
                # 0.9999999999993
                # NP:
                # 0.999999999999824
                # NPBAR:
                # 0.99999999999996
                # PP:
                # 1.00000000000053
                # S:
                # 0.9999999999997999
                # SBAR:
                # 0.9999999999999
                # SBARQ:
                # 1.0000000000001
                # SQ:
                # 0.9999999999999001
                # SQBAR:
                # 1.0000000000006
                # TOP:
                # 0.99999999999993
                # VPBAR:
                # 1.00000000000024
                # WHNP:
                # 1.0000000000003
                # X:
                # 0.9999999999989999
                # INTJ:
                # 0.9999999999999
                # else lhs prob_sum equal to 1.0



if __name__ == "__main__":
    # with open(sys.argv[1],'r') as grammar_file:
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        if grammar.verify_grammar():
            print("grammar is valid")
        else:
            print("grammar is invalid")

        
