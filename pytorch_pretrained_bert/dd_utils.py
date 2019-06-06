# coding=utf-8
import os
import sys
import csv
import math
import unicodedata
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import Counter

import torch
from torch import nn

HIGH_FIVE_REPLACEMENT = 'hit'

def normalize_string(string):
    """
    Normalizes the string from unicode to ascii,
    lowercasing and stripping whitespace
    """
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    return unicode_to_ascii(string.lower().strip())

def process_data(raw_path, proc_path, tokenizer, expt, nth_n):
    """
    Processes Tom McCoy's subj_aux dataset into the format, depending on experiment
    aux: <statement [as seq of ids]>\t<index of main auxiliary after BERT tokenization>
    sbjn:<statement [as seq of ids]>\t<index of subject noun after BERT tokenization>
    nth: <statement [as seq of ids]>\t<nth_n - 1>
    """

    # open file
    print(f"Processing: {raw_path}")
    with raw_path.open(mode='r') as f_raw, proc_path.open(mode='w') as f_proc:

        # read lines, normalize, and tokenize
        lines = f_raw.read().strip().split('\n')
        pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
        statements_tok = [pair[0] for pair in pairs]

        if expt == 'aux' or expt == 'sbjn':
            # labels from generation, processing step
            labels = [pair[1] for pair in pairs]
        elif expt == 'nth':
            labels = [nth_n - 1] * len(pairs)
        else:
            raise Exception(f"Unrecognized experiment type: {expt}")

        # write to f_proc
        for statement_tok, label in zip(statements_tok, labels):
            # compute the embeddings of tokens in the statement
            statement_ids = ' '.join(map(str, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(statement_tok))))
            f_proc.write(statement_ids + '\t' + str(label) + '\n')

def augment_data_JJ(x_data, y_data, data_raw, num_JJ, tokenizer):

    # error check
    assert(len(x_data) == len(y_data))
    assert(len(x_data) == len(data_raw))

    # duplicate
    import copy
    x_ret = tuple(copy.deepcopy(list(x_data)))
    y_ret = tuple(copy.deepcopy(list(y_data)))
    ret_raw = copy.deepcopy(data_raw)

    # adjectives
    JJs = ["small", "little", "big", "hot", "cold",
           "good", "bad", "new", "old", "young"]

    # insert adjectives
    np.random.seed(num_JJ)
    for i in range(len(ret_raw)):
        selected = np.random.choice(JJs, num_JJ, replace=True)
        statement = ret_raw[i].split('\t')[0].split(' ')
        question = ret_raw[i].split('\t')[1].split(' ')
        for JJ in selected:
            x_ret[i].insert(1, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(JJ))[0])
            statement.insert(1, JJ)
            question.insert(2, JJ)
        ret_raw[i] = ' '.join(statement) + '\t' + ' '.join(question) + '\n'

    y_ret = tuple([label+num_JJ for label in y_ret])

    assert(len(x_ret) == len(y_ret))
    assert(len(x_ret) == len(ret_raw))

    return x_ret, y_ret, ret_raw

def prep_batch(x_batch, y_batch, device):
    # pad all inputs to match the max-length input in the batch. We
    # temporarily use -1 to help us construct the mask, then replace it
    # with 0 once we are done because the transformer will
    # inadvertently try to index into the vocab with the id later.
    x_lengths = list(map(len, x_batch))
    x_batch = list(map(lambda x: torch.tensor(x).to(device), x_batch))
    x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=-1).to(device)

    # create one-hot vectors for labels
    y_batch = torch.tensor(y_batch).to(device)
    y_onehot = torch.zeros_like(x_batch).to(device)
    y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.)

    # create mask by constructing a tensor with rows [0, 1, 2, ... max_length-1]
    mask = torch.ones_like(x_batch).to(device)
    mask[x_batch < 0] = 0
    x_batch[x_batch < 0] = 0

    return x_batch, y_batch, y_onehot, mask
