#!/usr/bin/env/python

"""
Homework1 for CS 665
"""

import argparse
import os
import pprint
from collections import defaultdict, Counter
import os.path
import sys

import numpy as np

__authors__ = ['Dhananjay Singh <dsingh38@jhu.edu>', 'Kandarp Khandwala <kandarp.k@jhu,edu>',
               'Tom Mccoy <tom.mccoy@jhu.edu>']

'''
Grammar reading rules
nonterminal symbols, LHS, starting with
'''


class Grammar:
    IGNORE_LINE_PREFIXES = ['#']
    ROOT = 'ROOT'

    def __init__(self, file_name, enable_debug):

        self.rules = defaultdict(list)
        if not os.path.isfile(file_name):
            raise Exception('File not found at {}. Kindly check the command line arguments again'.format(file_name))

        with open(file_name) as f:
            for line in f:
                line = line.strip()

                # Ignore line if it is a comment (#)
                if not line or line[0] in Grammar.IGNORE_LINE_PREFIXES:
                    continue

                # Ignore any additional comments in the line
                line = line.split('#', 1)[0]

                parts = line.split()
                ok, reason = self.__validate__(parts)
                if len(parts) < 3:  # accomodate the condition where a relative rule is empty
                    parts.append('')
                if ok:
                    # Add to the rule set;  the probability and the LHS;
                    self.rules[parts[1]].append([float(parts[0]), parts[2:]])
                else:
                    print('Rule ignored {0} because {1}'.format(parts, reason))
            # Now that the grammar is loaded, calculate the probability of each rule
            self.__calc_prob__(self.rules, enable_debug)

    def __validate__(self, parts):
        # This will throw a value exception if its not valid number
        ok = True
        reason = ''
        if not isinstance(parts, list):
            ok, reason = False, 'Not a list'
        try:
            float(parts[0])  # first item should be a probability
            if float(parts[0]) < 0:
                ok, reason = False, 'Negative numbers aren\'t permitted in the first column of the grammar.'
        except ValueError:
            ok, reason = False, 'First item is not a number'

        return (ok, reason)

    def gen_rand_sent(self, enable_tree, enable_batch, enable_debug):
        rule_counter = Counter()

        #  Start at Root
        stack = [Grammar.ROOT]
        sentence = []
        just_sentence = []

        # Keep going till we run out of objects to fill, DFS style iteration
        while stack:
            current = stack.pop()

            # If terminal, add it to the sentence
            if enable_batch and current in ['S']:
                sentence.append('{')
                stack.extend('}')
            elif enable_batch and current in ['NP']:
                sentence.append('[')
                stack.extend(']')
            elif enable_tree and current in self.rules:
                sentence.append('(')
                sentence.append('{0}'.format(current))
                stack.extend(')')

            if current not in self.rules:
                sentence.append('{0}'.format(current))
                if current is not ')':
                    just_sentence.append(current)
                continue

            # If non terminal
            # Choose which rule to follow based on the probabilities from the grammar
            rule_pool = self.rules[current]
            selected_index = np.random.choice(len(rule_pool[1]), 1, p=rule_pool[0])
            rule = rule_pool[1][int(selected_index)]
            if not rule:
                # Empty rule
                break

            # sentence.append(current)
            stack.extend(reversed(rule))
            rule_counter[current] += 1


        if enable_debug:
            print "Constructed: ", ' '.join(just_sentence)


        # print(rule_counter.most_common(5))
        return ' '.join(sentence).replace('( ', '(').replace(' )', ')')

    def __calc_prob__(self, rules, enable_debug):
        # Calculate the probabilities and consolidate the weights and rhs
        for lhs in rules:
            weights, rhs = zip(*rules[lhs])
            total = sum(weights)
            weights = map(lambda w: w / total, weights)
            rules[lhs] = [weights, rhs]

        if enable_debug:
            pprint.pprint(dict(rules))


def positive_int(value):
    int_val = int(value)
    if int_val <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_val


def get_args():
    parser = argparse.ArgumentParser(description='Generate sentences from the grammar')
    parser.add_argument('file', help='Please provide a grammar file ', nargs='?')
    parser.add_argument('count', help='Please provide the number of sentences to generate ', type=positive_int,
                        nargs='?')
    parser.add_argument('-t', '--tree', help='Display the generated sentence in the form of a tree',
                        action='store_true', default=False)
    parser.add_argument('-b', '--batch', help='Display only sentences and noun phrases in a batch',
                        action='store_true', default=False)
    parser.add_argument('-d', '--debug', help='Display extra info', action='store_true', default=False)

    args = parser.parse_args()
    args.count = args.count or 1

    if not args.file or not os.path.isfile(args.file):
        raise argparse.ArgumentTypeError("Provide a valid grammar file ", args.file)

    return args


if __name__ == '__main__':
    args = get_args()
    grammar_file, num, tree, batch, debug = args.file, args.count, args.tree, args.batch, args.debug
    grammar = Grammar(grammar_file, enable_debug=debug)
    for i in xrange(num):
        print grammar.gen_rand_sent(enable_tree=tree, enable_batch=batch, enable_debug=debug)
