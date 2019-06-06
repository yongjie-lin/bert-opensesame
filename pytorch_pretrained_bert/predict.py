# coding=utf-8
import os
import sys
import csv
import math
import random
import time
import argparse
from itertools import count
import numpy as np

from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn

from pytorch_pretrained_bert import BertTokenizer, BertAdam, tokenization

from dd_models import BertForTokenClassificationLayered
from dd_utils import process_data, augment_data_JJ, prep_batch

TRAIN_FILE = '.train'
TEST_FILE = '.test'
GEN_FILE = '.gen'
MODEL_ABBREV = {'bbu': 'bert-base-uncased',
                'blu': 'bert-large-uncased',
                'bbmu': 'bert-base-multilingual-uncased',
}


def run_eval(model, x, y, tokenizer, device, batch_size=32, check_results=False):
    model.eval()
    with torch.no_grad():
        output_layers = model.module.output_layers
        layered_loss = [0. for _ in output_layers]
        layered_acc = [0. for _ in output_layers]
        layered_results = [[] for _ in output_layers]
        n = len(x)

        for start_idx in range(0, n, batch_size):
            x_batch = x[start_idx:start_idx+batch_size]
            y_batch = y[start_idx:start_idx+batch_size]
            x_batch, y_batch, y_onehot, mask = prep_batch(x_batch, y_batch, device)

            layered_loss_batch = model(x_batch, attention_mask=mask, labels=y_onehot)
            layered_logits = model(x_batch, attention_mask=mask)
            layered_preds = [torch.argmax(logits, dim=1).to(device) for logits in layered_logits]
            for idx, preds, loss_batch in zip(count(), layered_preds, layered_loss_batch):
                layered_acc[idx] += (preds[:,1] == y_batch).sum()
                layered_loss[idx] += loss_batch.sum()

                if check_results:
                    for batch_idx, correct in enumerate((preds[:,1] == y_batch).cpu().numpy()):
                        corpus_idx = start_idx + batch_idx
                        pred_idx = preds[batch_idx,1].item()
                        true_idx = y_batch[batch_idx].item()
                        pred_subword = x_batch[batch_idx][pred_idx].item()
                        true_subword = x_batch[batch_idx][true_idx].item()
                        pred_subword, true_subword = tokenizer.convert_ids_to_tokens([pred_subword, true_subword])
                        layered_results[idx].append((corpus_idx, correct, pred_subword, pred_idx, true_subword, true_idx))

    return [loss.item() for loss in layered_loss], [acc.item() / n for acc in layered_acc], layered_results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        choices=['train', 'eval'],
                        help="Specifies whether to run in train or eval (inference) mode.")

    parser.add_argument("--bert_model",
                        type=str,
                        required=True,
                        choices=['bbu', 'blu', 'bbmu'],
                        help="Bert pre-trained model to be used.")

    parser.add_argument("--expt",
                        type=str,
                        required=True,
                        choices=['aux', 'sbjn', 'nth'],
                        help="Type of experiment being conducted. Used to generate the labels.")

    # Other parameters
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training and inference.")

    parser.add_argument("--n_epochs",
                        default=8,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--nth_n",
                        default=None,
                        type=int,
                        help="Value of n (ONE-INDEXED) in nth experiments.")

    parser.add_argument("--output_layers",
                        default=[-1],
                        nargs='+',
                        type=int,
                        help="Space-separated list of (1 or more) layer indices whose embeddings will be used to train classifiers.")

    parser.add_argument("--eager_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run full evaluation (w/ generalization set) after each training epoch.")

    parser.add_argument("--load_checkpt",
                        default=None,
                        type=str,
                        help="Path to a checkpoint to be loaded, for training or inference.")

    parser.add_argument("--data_path",
                        default='data',
                        type=str,
                        help="Relative directory where train/test data is stored.")

    parser.add_argument("--expt_path",
                        default=None,
                        type=str,
                        help="Relative directory to store all data associated with current experiment.")

    args = parser.parse_args()
    bert_model = args.bert_model
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    data_path = Path(args.data_path)
    tokenizer = tokenization.BertTokenizer.from_pretrained(MODEL_ABBREV[bert_model], do_lower_case=True)

    global TRAIN_FILE, TEST_FILE, GEN_FILE
    TRAIN_FILE = args.expt + TRAIN_FILE
    TEST_FILE = args.expt + TEST_FILE
    GEN_FILE = args.expt + GEN_FILE

    # error check
    if args.mode == 'eval' and args.load_checkpt is None:
        raise Exception(f"{__file__}: error: the following arguments are required in eval mode: --load-checkpt")
    if args.expt == 'nth' and args.nth_n is None:
        raise Exception(f"{__file__}: error: the following arguments are required in nth expts: --nth_n")

    # experiment dir
    Path('experiments').mkdir(exist_ok=True)
    if args.expt_path is None:
        expt_path = Path("experiments/{0}_{1}".format(args.expt, time.strftime("%Y%m%d-%H%M%S")))
    else:
        expt_path = Path(args.expt_path)
    expt_path.mkdir(exist_ok=True)

    # cuda
    print('Initializing...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()

    # if processed data doesn't exist, create it
    for filename in (TRAIN_FILE, TEST_FILE, GEN_FILE):
        if (expt_path / f"{filename}.{bert_model}.proc").exists():
            print(f"Found {expt_path}/{filename}.{bert_model}.proc")
        elif (data_path / filename).exists():
            process_data(data_path / filename, expt_path / f"{filename}.{bert_model}.proc", tokenizer, args.expt, args.nth_n)
        else:
            raise FileNotFoundError(f"{data_path / filename} not found! Download from https://github.com/tommccoy1/subj_aux/tree/master/data")

    # load processed data
    train_path, test_path, gen_path = (expt_path / f"{filename}.{bert_model}.proc" for filename in (TRAIN_FILE, TEST_FILE, GEN_FILE))
    with train_path.open() as train, test_path.open() as test, gen_path.open() as gen:
        xy_train = ([int(tok) for tok in str.split(line)] for line in train)
        x_train, y_train = zip(*((line[:-1], line[-1]) for line in xy_train))
        xy_test = ([int(tok) for tok in str.split(line)] for line in test)
        x_test, y_test = zip(*((line[:-1], line[-1]) for line in xy_test))
        xy_gen = ([int(tok) for tok in str.split(line)] for line in gen)
        x_gen, y_gen = zip(*((line[:-1], line[-1]) for line in xy_gen))
    n_train, n_test, n_gen = len(x_train), len(x_test), len(x_gen)

    # initialize BERT model
    model = BertForTokenClassificationLayered.from_pretrained(MODEL_ABBREV[bert_model], output_layers=args.output_layers)
    model.to(device)

    # distribute model over GPUs, if available
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer: freeze all BERT weights
    for name, param in model.named_parameters():
        param.requires_grad = bool('classifier' in name)
    param_optimizer = [p for p in model.named_parameters() if 'classifier' in p[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer
    optimizer = BertAdam(optimizer_grouped_parameters, lr=0.001)

    # load from past checkpoint if given
    if args.load_checkpt is not None:
        print(f"Loading from {args.load_checkpt}...")
        checkpt = torch.load(Path(args.load_checkpt))
        epoch = checkpt['epoch']
        model_dict = model.state_dict()
        model_dict.update(checkpt['model_state_dict'])
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    else:
        epoch = 0

    if args.mode == 'train':
        train_msg = f"""Starting training...
            - # GPUs available: {n_gpu}
            - BERT model: {MODEL_ABBREV[bert_model]}
            - Experiment: {args.expt}{', n=' + str(args.nth_n) if args.nth_n is not None else ''}
            - Batch size: {batch_size}
            - # training epochs: {n_epochs}
            - Output layers: {args.output_layers}
        """
        print(train_msg)

        layered_best_valid_loss = [float('inf') for _ in args.output_layers]
        for _ in range(n_epochs):
            epoch += 1
            model.train()
            layered_running_loss = [0. for _ in args.output_layers]
            for batch_num in range(1, n_train // batch_size+1):
                with torch.no_grad():
                    idx = random.choices(range(n_train), k=batch_size)
                    x_batch = [x_train[i] for i in idx]
                    y_batch = [y_train[j] for j in idx]
                    x_batch, _, y_onehot, mask = prep_batch(x_batch, y_batch, device)

                optimizer.zero_grad()
                layered_loss_batch = model(x_batch, attention_mask=mask, labels=y_onehot)
                for idx, loss_batch in enumerate(layered_loss_batch):
                    loss = loss_batch.sum()
                    loss.backward()
                    layered_running_loss[idx] += loss
                optimizer.step()

                if batch_num % 500 == 0:
                    for output_layer, running_loss in zip(args.output_layers,layered_running_loss):
                        layer_str = f"Layer[{output_layer}]" if args.output_layers != [-1] else ''
                        print(f"Epoch {epoch} ({batch_num * batch_size}/{n_train})", layer_str, f"loss: {running_loss / (batch_num  * batch_size):.7f}", flush=True)

            # compute validation loss and accuracy
            layered_valid_loss, layered_valid_acc, layered_valid_results = run_eval(model, x_test, y_test, tokenizer, device, batch_size=batch_size, check_results=True)

            print(f"END of epoch {epoch}, saving checkpoint...")

            for idx, output_layer, running_loss, valid_loss, valid_acc, valid_results in zip(count(), args.output_layers, layered_running_loss, layered_valid_loss, layered_valid_acc, layered_valid_results):
                layer_str = f"Layer[{output_layer}]" if args.output_layers != [-1] else ''
                valid_msg = f"Validation loss: {valid_loss:.7f}, acc: {valid_acc:.7f}"
                layer_idx_str = f"[{output_layer}]" if args.output_layers != [-1] else ''
                results_str = f"{bert_model}{layer_idx_str}.results"
                checkpt_str = f"{bert_model}{layer_idx_str}_e{epoch}_l{valid_loss:8f}.ckpt"

                print(layer_str, valid_msg, flush=True)
                with (data_path / f"{TEST_FILE}").open() as test:
                    test_raw = test.readlines()
                # write results to file
                with (expt_path / f"{TEST_FILE}.{results_str}").open(mode='w') as f_valid:
                    for t_idx, result, pred_subword, pred_idx, true_subword, true_idx in valid_results:
                        f_valid.write(f"#{t_idx}\t{result}\tPrediction: {pred_subword} ({pred_idx})\tTrue: {true_subword} ({true_idx})\t{test_raw[t_idx]}\n")

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': {param: value for param, value in model.state_dict().items() if 'classifier' in param},
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': layered_running_loss,
                            'valid_loss': layered_valid_loss,
                }, expt_path / checkpt_str)

                if valid_loss < layered_best_valid_loss[idx]:
                    layered_best_valid_loss[idx] = valid_loss
                    print(layer_str, f"best model is {checkpt_str}", flush=True)

                    best_str = f"{bert_model}{layer_idx_str}_best.ckpt"
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': {param: value for param, value in model.state_dict().items() if 'classifier' in param},
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': layered_running_loss,
                                'valid_loss': layered_valid_loss,
                    }, expt_path / best_str)
            if args.eager_eval:
                # compute gen loss and accuracy
                layered_gen_loss, layered_gen_acc, layered_gen_results = run_eval(model, x_gen, y_gen, tokenizer, device, batch_size=batch_size, check_results=True)

                for idx, output_layer, gen_loss, gen_acc, gen_results in zip(count(), args.output_layers, layered_gen_loss, layered_gen_acc, layered_gen_results):
                    layer_str = f"Layer[{output_layer}]" if args.output_layers != [-1] else ''
                    gen_msg = f"Generalization loss: {gen_loss:.7f}, acc: {gen_acc:.7f}"
                    layer_idx_str = f"[{output_layer}]" if args.output_layers != [-1] else ''
                    results_str = f"{bert_model}{layer_idx_str}.results"

                    print(layer_str, gen_msg, flush=True)
                    with (data_path / f"{GEN_FILE}").open() as gen:
                        gen_raw = gen.readlines()
                    # write results to file
                    with (expt_path / f"{GEN_FILE}.{results_str}").open(mode='w') as f_gen:
                        for g_idx, result, pred_subword, pred_idx, true_subword, true_idx in gen_results:
                            f_gen.write(f"#{g_idx}\t{result}\tPrediction: {pred_subword} ({pred_idx})\tTrue: {true_subword} ({true_idx})\t{gen_raw[g_idx]}\n")

    elif args.mode == 'eval':

        eval_msg = f"""Starting evaluation...
            - # GPUs available: {n_gpu}
            - Checkpoint: {args.load_checkpt}
            - BERT model: {MODEL_ABBREV[bert_model]}
            - Experiment: {args.expt}{', n=' + str(args.nth_n) if args.nth_n is not     None else ''}
            - Batch size: {batch_size}
            - # training epochs: {n_epochs}
            - Output layers: {args.output_layers}
        """
        print(eval_msg)

        test_raw_path, gen_raw_path = (data_path / f"{filename}" for filename in (TEST_FILE, GEN_FILE))
        with test_raw_path.open() as test, gen_raw_path.open() as gen:
            test_raw = test.readlines()
            gen_raw = gen.readlines()

        # compute validation loss and accuracy for test and gen
        layered_valid_loss, layered_valid_acc, layered_valid_results = run_eval(model, x_test, y_test, tokenizer, device, check_results=True)
        layered_gen_loss, layered_gen_acc, layered_gen_results = run_eval(model, x_gen, y_gen, tokenizer, device, check_results=True)

        for output_layer, valid_loss, valid_acc, valid_results, gen_loss, gen_acc, gen_results in zip(args.output_layers, layered_valid_loss, layered_valid_acc, layered_valid_results, layered_gen_loss, layered_gen_acc, layered_gen_results):
            layer_str = f"Layer[{output_layer}]" if args.output_layers != [-1] else ''
            layer_idx_str = f"[{output_layer}]" if args.output_layers != [-1] else ''
            valid_msg = f"Validation loss: {valid_loss:.7f}, acc: {valid_acc:.7f}"
            gen_msg = f"Generalization loss: {gen_loss:.7f}, acc: {gen_acc:.7f}"
            results_str = f"{bert_model}{layer_idx_str}.results"

            for msg in (valid_msg, gen_msg):
                print(layer_str, msg)

            # write results to file
            with (expt_path / f"{TEST_FILE}.{results_str}").open(mode='w') as f_valid, (expt_path / f"{GEN_FILE}.{results_str}").open(mode='w') as f_gen:
                for idx, result, pred_subword, pred_idx, true_subword, true_idx in valid_results:
                    f_valid.write(f"#{idx}\t{result}\tPrediction: {pred_subword} ({pred_idx})\tTrue: {true_subword} ({true_idx})\t{test_raw[idx]}\n")

                for idx, result, pred_subword, pred_idx, true_subword, true_idx in gen_results:
                    f_gen.write(f"#{idx}\t{result}\tPrediction: {pred_subword} ({pred_idx})\tTrue: {true_subword} ({true_idx})\t{gen_raw[idx]}\n")

    else:
        raise Exception('--mode must be set to either "train" or "eval"')

if __name__ == '__main__':
    main()
