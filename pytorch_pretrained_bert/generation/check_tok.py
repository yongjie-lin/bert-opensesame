# coding=utf-8
import os
import sys
import csv
import math
import random
import time
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn

from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer, BertAdam, tokenization

TRAIN_FILE = 'no_agr.train'
TEST_FILE = 'no_agr.test'
GEN_FILE = 'no_agr.gen'
MODEL_ABBREV = {'bbu': 'bert-base-uncased',
                'blu': 'bert-large-uncased',
                'bbmu': 'bert-base-multilingual-uncased',
}

def main():

    # aux tokens
    aux_tokens = {"the", "some", "my", "your", "our", "her",
                  "bird", "bee", "ant", "duck", "lion", "dog", "tiger", "worm", "horse", "cat", "fish", "bear", "wolf",
                  "birds", "bees", "ants", "ducks", "lions", "dogs", "tigers", "worms", "horses", "cats", "fish", "bears", "wolves",
                  "cry", "smile", "sleep", "swim", "wait", "move",  "change", "read", "eat",
                  "dress", "kick", "hit", "hurt", "clean", "love", "accept", "remember", "comfort",
                  "can", "will", "would", "could",
                  "can", "will", "would", "could",
                  "around", "near", "with", "upon", "by", "behind", "above", "below",
                  "who", "that",
                  "small", "little", "big", "hot", "cold", "good", "bad", "new", "old",  "young"}

    # sbjn tokens
    sbjn_tokens = {"the", "some", "my", "your", "our", "her",
                   "'s",
                   "bird", "bee", "ant", "duck", "lion", "dog", "tiger", "worm", "horse", "cat", "fish", "bear", "wolf",
                   "bird", "bee", "ant", "duck", "lion", "dog", "tiger", "worm", "horse", "cat", "fish", "bear", "wolf",
                   "birds", "bees", "ants", "ducks", "lions", "dogs", "tigers", "worms", "horses", "cats", "fish", "bears", "wolves",
                   "worker", "ant",
                   "worker", "bee",
                   "german", "dog",
                   "house", "cat",
                   "bird", "bee", "ant", "duck", "lion", "dog", "tiger", "worm", "horse", "cat", "fish", "bear", "wolf",
                   "birds", "bees", "ants", "ducks", "lions", "dogs", "tigers", "worms", "horses", "cats", "fish", "bears", "wolves",
                   "cry", "smile", "sleep", "swim", "wait", "move",  "change", "read", "eat",
                   "dress", "kick", "hit", "hurt", "clean", "love", "accept", "remember", "comfort",
                   "can", "will", "would", "could",
                   "around", "near", "with", "upon", "by", "behind", "above", "below",
                   "who", "that",
                   "small", "little", "big", "hot", "cold", "good", "bad", "new", "old",  "young"}

    # refl tokens
    refl_tokens = {"the", "some", "my", "your", "our", "her",
                   "girl", "woman", "queen", "actress", "sister", "wife", "mother", "princess", "aunt", "lady", "witch", "niece", "nun",
                   "boy", "man", "king", "actor", "brother", "husband", "father", "prince", "uncle", "lord", "wizard", "nephew", "monk",
                   "cry", "smile", "sleep", "swim", "wait", "move",  "change", "read", "eat",
                   "dress", "kick", "hit", "hurt", "clean", "love", "accept", "remember", "comfort",
                   "think", "say", "hope", "know",
                   "tell", "convince", "persuade", "inform",
                   "can", "will", "would", "could",
                   "around", "near", "with", "upon", "by", "behind", "above", "below",
                   "who", "that"}

    # agr tokens
    agr_tokens = {"the", "my", "your", "our", "her",
                  "bird", "bee", "ant", "duck", "lion", "dog", "tiger", "worm", "horse", "cat", "fish", "bear", "wolf",
                  "birds", "bees", "ants", "ducks", "lions", "dogs", "tigers", "worms", "horses", "cats", "fish", "bears", "wolves",
                  "cry", "smile", "sleep", "swim", "wait", "move",  "change", "read", "eat",
                  "dress", "kick", "hit", "hurt", "clean", "love", "accept", "remember", "comfort",
                  "think", "say", "hope", "know",
                  "tell", "convince", "persuade", "inform",
                  "does", "do",
                  "can", "will", "would", "could",
                  "around", "near", "with", "upon", "by", "behind", "above", "below",
                  "who", "that"}

    # misc tokens
    misc_tokens = {"'s"}

    # test tokens
    for bert_model in ["bbu", "blu", "bbmu"]:
        change = set()
        print(MODEL_ABBREV[bert_model])
        tokenizer = tokenization.BertTokenizer.from_pretrained(MODEL_ABBREV[bert_model], do_lower_case=True)

        # aux tokens
        for tok in list(aux_tokens):
            sentence = "%s" % tok
            tokenized = tokenizer.tokenize(sentence)
            for i, token in enumerate(tokenized):
                if '#' in token:
                    change.add(tokenized[i-1] + tokenized[i])

        # sbjn tokens
        for tok in list(sbjn_tokens):
            sentence = "%s" % tok
            tokenized = tokenizer.tokenize(sentence)
            for i, token in enumerate(tokenized):
                if '#' in token:
                    change.add(tokenized[i-1] + tokenized[i])

        # refl tokens
        for tok in list(refl_tokens):
            sentence = "%s" % tok
            tokenized = tokenizer.tokenize(sentence)
            for i, token in enumerate(tokenized):
                if '#' in token:
                    change.add(tokenized[i-1] + tokenized[i])

        # agr tokens
        for tok in list(agr_tokens):
            sentence = "%s" % tok
            tokenized = tokenizer.tokenize(sentence)
            for i, token in enumerate(tokenized):
                if '#' in token:
                    change.add(tokenized[i-1] + tokenized[i])

        # agr tokens
        for tok in list(misc_tokens):
            sentence = "%s" % tok
            tokenized = tokenizer.tokenize(sentence)
            print(tokenized)
            for i, token in enumerate(tokenized):
                if '#' in token:
                    change.add(tokenized[i-1] + tokenized[i])

        print("change", change)
        print()

if __name__ == '__main__':
    main()
