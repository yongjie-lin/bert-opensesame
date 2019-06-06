# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Change log
# 12/12/18  Jesse Vig   Adapted to BERT model
# 12/19/18  Jesse Vig   Assorted cleanup. Changed orientation of attention matrices. Updated comments.


"""Module for postprocessing and displaying transformer attentions.

This module is designed to be called from an ipython notebook.
"""

import json
import os
import numpy as np

import IPython.display as display


vis_html = """
  <span style="user-select:none">
    Layer: <select id="layer"></select>
    Attention: <select id="att_type">
      <option value="all">All</option>
      <option value="a">Sentence A self-attention</option>
      <option value="b">Sentence B self-attention</option>
      <option value="ab">Sentence A -> Sentence B</option>
      <option value="ba">Sentence B -> Sentence A</option>
      <option value="avg_aa">Head-averaged sentence A self-attention</option>
      <option value="up2k_aa">Reduced up-to-k sentence A self-attention</option>
    </select>
  </span>
  <div id='vis'></div>
"""

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
vis_js = open(os.path.join(__location__, 'attention.js')).read()


def show(tokens_a, tokens_b, attn, expt_params):
    """Displays attention visualization.
    expt_params: a dictionary possibly containing the following keys
        token_groups: array of nonnegative integers indicating how tokens are to be grouped in the viz
            e.g. "The quick brown fox jumps over the lazy dog ." with [1,1,1,1,0,0,2,2,2,0] produces
            the target groups "The quick brown fox" and "the lazy dog". ONLY WORKS WITH A->A, B->A FOR NOW.
        attn_sources: array of indices of the tokens (in tokens_a) with attention distributions that we are interested in.
        attn_target_groups: array of indices corresponding to token group that the self-attention of the corresponding
            source should be focusing on, for the purpose of computing binary cross-entropy. Only allowed values are 1
            and 2.
    """

    params = expt_params.keys()
    if 'token_groups' in params:
        token_groups = expt_params['token_groups']
        assert(len(token_groups) == len(tokens_a) - 2)
        assert(all(type(i) is int for i in token_groups))

    if 'attn_sources' in params and 'attn_target_groups' in params:
        attn_sources = expt_params['attn_sources']
        attn_target_groups = expt_params['attn_target_groups']
        assert(set(token_groups) == set([0,1,2]))
        assert(len(attn_sources) == len(attn_target_groups))
        assert(set(attn_sources).issubset(set(range(len(tokens_a)))))
        assert(set(attn_target_groups).issubset(set([1,2])))
    elif 'attn_sources' in params or 'attn_target_groups' in params:
        raise ValueError('Please provide both attn_sources and attn_target_groups, otherwise omit both of them.')

    attentions = _get_attentions(tokens_a, tokens_b, attn, expt_params)
    att_json = json.dumps(attentions)
    _show_attention(att_json)


def _show_attention(att_json):
    display.display(display.HTML(vis_html))
    display.display(display.Javascript('window.attention = %s' % att_json))
    display.display(display.Javascript(vis_js))


def logmatmulexp(A, B): # assuming A,B have shape [1, n, n]
    max_A = np.max(A, -1, keepdims=True)
    max_B = np.max(B, -1, keepdims=True)
    C = np.matmul(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + np.transpose(max_B, (0,2,1))
    return C

def _get_attentions(tokens_a, tokens_b, attn, expt_params):
    """Compute representation of the attention to pass to the d3 visualization

    Args:
      tokens_a: tokens in sentence A
      tokens_b: tokens in sentence B
      attn: numpy array, attention
          [num_layers, batch_size, num_heads, seq_len, seq_len]
      expt_params: dictionary containing customizations for the viz, e.g. target groups and inputs for
          computing cross-entropy

    Returns:
      Dictionary of attention representations with the structure:
      {
        'all': Representations for showing all attentions at the same time. (source = AB, target = AB)
        'a': Sentence A self-attention (source = A, target = A)
        'b': Sentence B self-attention (source = B, target = B)
        'ab': Sentence A -> Sentence B attention (source = A, target = B)
        'ba': Sentence B -> Sentence A attention (source = B, target = A)
      }
      and each sub-dictionary has structure:
      {
        'att': list of inter attentions matrices, one for each layer. Each is of shape [num_heads, source_seq_len, target_seq_len]
        'top_text': list of source tokens, to be displayed on the left of the vis
        'bot_text': list of target tokens, to be displayed on the right of the vis
      }
    """

    all_attns = []
    a_attns = []
    b_attns = []
    ab_attns = []
    ba_attns = []
    slice_a = slice(0, len(tokens_a)) # Positions corresponding to sentence A in input
    slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b)) # Position corresponding to sentence B in input

    avg_attns = []
    up2k_attns = []
    # up2k = np.expand_dims(np.identity(len(tokens_a)), 0) # initialize accumulator for reduction operation
    log_up2k = None
    tokens_a_grouped = None
    no_sep_slice = slice(1, len(tokens_a)-1) # for renormalization so viz is not dominated by [CLS], [SEP] attentions

    if 'token_groups' in expt_params.keys():
        token_groups = expt_params['token_groups']
        token_groups.insert(0, 0) # add 0 for [CLS]
        token_groups.append(0) # add 0 for [SEP]
        d = {i: [idx for (idx,grp) in enumerate(token_groups) if grp == i] for i in set(token_groups)}
        tokens_a_grouped = []
        for grp, idx_list in d.items():
            if grp == 0:
                continue
            tokens_a_grouped.append(' '.join(tokens_a[idx] for idx in idx_list))
        print("Token groups:", list(enumerate(tokens_a_grouped, 1)))
    else:
        print('Number of tokens:', len(tokens_a))
        token_groups = None

    head_visual_scaling_factor = 1
    up2k_visual_scaling_factor = 1
    num_layers = len(attn)
    for layer in range(num_layers):
        layer_attn = attn[layer][0] # Get layer attention (assume batch size = 1), shape = [num_heads, seq_len, seq_len]
        all_attns.append(layer_attn.tolist()) # Append AB->AB attention for layer, across all heads
        b_attns.append(layer_attn[:, slice_b, slice_b].tolist()) # Append B->B attention for layer, across all heads
        ab_attns.append(layer_attn[:, slice_a, slice_b].tolist()) # Append A->B attention for layer, across all heads

        aa_attn = layer_attn[:, slice_a, slice_a] # keep only the a->a attentions
        aa_attn /= aa_attn.sum(axis=2, keepdims=True) # renormalize axis 2 of aa_attn after slicing
        head_avg = np.mean(aa_attn, axis=0, keepdims=True) # mean preserves normalization along axis 2

        # normalizer = head_avg[:, :, no_sep_slice].sum(axis=2, keepdims=True)
        # avg_attns.append((head_visual_scaling_factor * head_avg / normalizer).tolist())

        if log_up2k is None:
            log_up2k = np.log(head_avg)
        else:
            log_head_avg = np.log(head_avg)
            log_up2k = logmatmulexp(log_head_avg, log_up2k) # more numerically stable than chaining matmuls

        # np.matmul(head_avg, up2k, out=up2k)
        # up2k /= up2k.sum(axis=2, keepdims=True)
        # normalizer = np.exp(log_up2k)[:, :, no_sep_slice].sum(axis=2, keepdims=True)
        # up2k_attns.append((up2k_visual_scaling_factor * np.exp(up2k) / normalizer).tolist())

        if token_groups is not None:
            a_attn_grouped = None
            ba_attn_grouped = None
            avg_attn_grouped = None
            up2k_attn_grouped = None
            for grp, idx_list in d.items():
                if grp == 0: # group 0 only consists of ignored tokens
                    continue
                if a_attn_grouped is None: # first iter
                    a_attn_grouped = layer_attn[:, slice_a, idx_list].sum(axis=2, keepdims=True)
                    ba_attn_grouped = layer_attn[:, slice_b, idx_list].sum(axis=2, keepdims=True)
                    avg_attn_grouped = head_avg[:, slice_a, idx_list].sum(axis=2, keepdims=True)
                    up2k_attn_grouped = np.exp(log_up2k)[:, slice_a, idx_list].sum(axis=2, keepdims=True)
                else:
                    a_attn_grouped = np.append(a_attn_grouped, layer_attn[:, slice_a, idx_list].sum(axis=2, keepdims=True), axis=2)
                    ba_attn_grouped = np.append(ba_attn_grouped, layer_attn[:, slice_b, idx_list].sum(axis=2, keepdims=True), axis=2)
                    avg_attn_grouped = np.append(avg_attn_grouped, head_avg[:, slice_a, idx_list].sum(axis=2, keepdims=True), axis=2)
                    up2k_attn_grouped = np.append(up2k_attn_grouped, np.exp(log_up2k)[:, slice_a, idx_list].sum(axis=2, keepdims=True), axis=2)
            a_attns.append(a_attn_grouped.tolist()) # Append A->A attention for layer, across all heads
            ba_attns.append(ba_attn_grouped.tolist()) # Append B->A attention for layer, across all heads
            normalizer = avg_attn_grouped.sum(axis=2, keepdims=True)
            avg_attns.append((head_visual_scaling_factor * avg_attn_grouped / normalizer).tolist())
            normalizer = up2k_attn_grouped.sum(axis=2, keepdims=True)
            up2k_attns.append((up2k_visual_scaling_factor * up2k_attn_grouped / normalizer).tolist())
        else:
            a_attns.append(layer_attn[:, slice_a, slice_a].tolist()) # Append A->A attention for layer, across all heads
            ba_attns.append(layer_attn[:, slice_b, slice_a].tolist()) # Append B->A attention for layer, across all heads
            normalizer = head_avg[:, :, no_sep_slice].sum(axis=2, keepdims=True)
            avg_attns.append((head_visual_scaling_factor * head_avg / normalizer).tolist())
            normalizer = np.exp(log_up2k)[:, :, no_sep_slice].sum(axis=2, keepdims=True)
            up2k_attns.append((up2k_visual_scaling_factor * np.exp(log_up2k) / normalizer).tolist())

    if 'attn_sources' in expt_params.keys():
        attn_sources, attn_target_groups = expt_params['attn_sources'], expt_params['attn_target_groups']
        print(f"{'Attention source':<20}{'Target group':<20}{'Binary cross-entropy'}")
        for idx in range(len(attn_sources)):
            source_idx = attn_sources[idx]
            target_group = attn_target_groups[idx]
            attn_vector = np.array(avg_attns)[:, 0, source_idx, target_group-1]

            # since bce(y,y*) = - y*log(y) - (1-y*)log(1-y) and we have y* = 1 in our use case
            bce = - np.log(attn_vector).sum()
            print(f"{tokens_a[source_idx]:<20}{tokens_a_grouped[target_group - 1]:<20}{bce:.5f}")

    attentions =  {
        'all': {
            'att': all_attns,
            'top_text': tokens_a + tokens_b,
            'bot_text': tokens_a + tokens_b
        },
        'a': {
            'att': a_attns,
            'top_text': tokens_a,
            'bot_text': tokens_a if token_groups is None else tokens_a_grouped
        },
        'b': {
            'att': b_attns,
            'top_text': tokens_b,
            'bot_text': tokens_b
        },
        'ab': {
            'att': ab_attns,
            'top_text': tokens_a,
            'bot_text': tokens_b
        },
        'ba': {
            'att': ba_attns,
            'top_text': tokens_b,
            'bot_text': tokens_a if token_groups is None else tokens_a_grouped
        },
        'avg_aa': {
            'att': avg_attns,
            'top_text': tokens_a,
            'bot_text': tokens_a if token_groups is None else tokens_a_grouped
        },
        'up2k_aa': {
            'att': up2k_attns,
            'top_text': tokens_a,
            'bot_text': tokens_a if token_groups is None else tokens_a_grouped
        },
    }
    return attentions
