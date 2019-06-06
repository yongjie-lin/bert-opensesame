from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from file_utils import cached_path

import pytorch_pretrained_bert.modeling as ppbm

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
logger = logging.getLogger(__name__)


class LinearAfterDropout(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out = nn.Linear(config.hidden_size, num_labels)

    def forward(self, inputs):
        return self.out(self.dropout(inputs))


class BertForTokenClassificationLayered(ppbm.PreTrainedBertModel):
    def __init__(self, config, num_labels=2, output_layers=[-1]):
        super(BertForTokenClassificationLayered, self).__init__(config)
        self.num_labels = num_labels
        self.bert = ppbm.BertModel(config)
        self.output_layers = output_layers
        self.classifiers = nn.ModuleList([LinearAfterDropout(config, num_labels) for _ in output_layers])
        self.apply(self.init_bert_weights)

    @property
    def output_layers(self):
        return self.__output_layers

    @output_layers.setter
    def output_layers(self, layers):
        assert(len(layers) == len(set(layers)))
        assert(set(layers).issubset(set(range(-self.bert.config.num_hidden_layers, self.bert.config.num_hidden_layers))))
        self.__output_layers = layers

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        layered_logits = [classifier(sequence_output[idx]) for classifier, idx in zip(self.classifiers, self.output_layers)]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                layered_active_logits = [logits.view(-1, self.num_labels)[active_loss] for logits in layered_logits]
                active_labels = labels.view(-1)[active_loss]
                layered_loss = [loss_fct(active_logits, active_labels) for active_logits in layered_active_logits]
            else:
                layered_loss = [loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) for logits in layered_logits]
            return layered_loss
        else:
            return layered_logits
