import sys
import numpy as np

import torch
from torch import nn

from tqdm import tqdm

from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer, BertAdam, tokenization

MODEL_ABBREV = {'bbu': 'bert-base-uncased',
                'blu': 'bert-large-uncased',
                'bbmu': 'bert-base-multilingual-uncased',
}

# REFL
    # transitive with simple NP
    # transitive with complex NP
        # intervening same gender
        # intervening diff gender
    # ambiguous cases (AMB)
    # structurally prior cases
        # VS case
            # intervening same gender
            # intervening diff gender
        # VD case
            # intervening same gender
            # intervening diff gender
        # AdvP case
            # intervening same gender
            # intervening diff gender


    # no rel cl
        # 0 no NP_o
        # 1 NP_o matches
        # 2 NP_o mismatches
    # rel cl
        # no NP_o
            # 0 no NP_r
            # 3 NP_r matches
            # 4 NP_r mismatches
        # NP_o matches
            # 5 NP_r matches
            # 6 NP_r mismatches
        # NP_o mis matcches
            # 7 NP_r matches
            # 8 NP_r mismatches

# AGR
    # simple subject
        # modal Aux
        # non-modal Aux
    # complex subject
        # intervening same number
            # modal Aux
            # non-modal Aux
        # intervening diff number
            # modal Aux
            # non-modal Aux
    # structurally prior cases
        # VS case
            # intervening same number
            # intervening diff number
        # VD case
            # intervening same number
            # intervening diff number
        # AdvP case
            # intervening same number
            # intervening diff number

    # normal
        # sbj s
            # 0 m
            # 1 nm
        # sbj p
            # 2 m
            # 3 nm
    # Prep
        # sbj s
            # NP_p s
                # 4 m
                # 5 nm
            # NP_p p
                # 6 m
                # 7 nm
        # sbj p
            # NP_p s
                # 8 m
                # 9 nm
            # NP_p p
                # 10 m
                # 11 nm
    # Rel
        # sbj s
            # no NP_r
                # 0 m
                # 1 nm
            # NP_r s
                # 12 m
                # 13 nm
            # NP_r p
                # 14 m
                # 15 nm
        # sbj p
            # no NP_r
                # 2 m
                # 3 nm
            # NP_r s
                # 16 m
                # 17 nm
            # NP_r p
                # 18 m
                # 19 nm

    # Subj is  plural or  singular. sinngular main sbj on plural  distractor => more likyl to be distracted with plural verb
    # If subj is plural, distractor snigular, ppl wont have problems
    #
    # type 1,2
    # type 5,6

    # always true for same type of sentences

    #  PP
    # modal  S rel
    # model O rel
    #  agr  S  rel
    # agr O rel

# dill

    # agr
    # NP_r
        # 0 s
        # 1 p

    # refl
    # NP_s s
        # 2 NP_r s
        # 3  NP_r p
    # NP_s p
        # 4 NP_r s
        # 5 NP_r p
prefix = sys.argv[1]
# functions
def get_sentence_state(idxs, ref_idx, prefix):

    if prefix == "aux":
        for idx in idxs:
            if idx < ref_idx:
                return "Rel_NP_M"
            elif idx > ref_idx:
                return "Rel_NP_O"
        return "No_Rel"

    elif prefix == "sbjn":
        for idx in idxs:
            if idx < ref_idx:
                return "Intervening"
            elif idx > ref_idx:
                return "Non Intervening"

def open_files(prefix, n, pos):

    files = []
    for i in range(n):
        if pos:
            f = open(f"{prefix}.{i}.pos", "w")
        else:
            f = open(f"{prefix}.{i}", "w")
        files.append(f)

    return files

def close_files(files):

    for file in files:
        file.close()

# if generating for aux expt:
if prefix == "aux":

    used_dict = {}
    fi = open("raw." + prefix, "r")

    fo_train = open(prefix +  ".train", "w")
    fo_test = open(prefix + ".test", "w")
    fo_gen = open(prefix + ".gen", "w")

    fo_train_POS = open(prefix +  ".train.pos", "w")
    fo_test_POS = open(prefix + ".test.pos", "w")
    fo_gen_POS = open(prefix + ".gen.pos", "w")

    count_train = 0
    count_test = 0
    count_gen = 0

    # initialize variables
    train_size = 40000
    test_size = 10000
    gen_size = 10000

    # process raw line by line
    for i, line in enumerate(fi):

        # if done with writing files, break
        if count_train >= train_size and count_test >= test_size and count_gen >= gen_size:
            break

        # if sentence seen, continue
        line = line.strip()
        if line in used_dict:
            continue

        # remember seen
        used_dict[line] = 1

        # tokenize into (word, POS)
        tokens = line.split(' ')[:-1]
        tokens = [(token.split('/')[0], token.split('/')[1]) for token in tokens]
        words = [token[0] for token in tokens]
        POSs = [token[1] for token in tokens]

        # find index of MAux
        maux_idx = POSs.index("MAux")

        # determine if sentence is Rel_NP_M or Rel_NP_O or No_Rel
        # Rel occuring before MAux is on NP_M, Rel occuring after MAux is on NP_O
        rel_idxs = [i for i, POS in enumerate(POSs) if POS == "Rel"]
        sentence_state = get_sentence_state(rel_idxs, maux_idx)

        # join words and POSs
        result_words = " ".join(words) + "\t" + str(maux_idx) + "\n"
        result_POSs = " ".join(POSs) + "\t" + str(maux_idx) + "\n"

        # write to corresponding file
        if sentence_state == "Rel_NP_M":
            if count_gen >= gen_size:
                continue
            fo_gen.write(result_words)
            fo_gen_POS.write(result_POSs)
            count_gen += 1
        else:
            coin = np.random.choice([0,1], size=1, p=[0.5,0.5])
            if coin:
                if count_test >= test_size:
                    continue
                fo_test.write(result_words)
                fo_test_POS.write(result_POSs)
                count_test += 1
            else:
                if count_train >= train_size:
                    continue
                fo_train.write(result_words)
                fo_train_POS.write(result_POSs)
                count_train += 1

    # print length of files written
    print(f"train: {count_train}")
    print(f"test: {count_test}")
    print(f"gen: {count_gen}")

    fo_train.close()
    fo_test.close()
    fo_gen.close()
    fo_train_POS.close()
    fo_test_POS.close()
    fo_gen_POS.close()

# else if generating for sbjn expt:
elif prefix == "sbjn":

    used_dict = {}
    fi = open("raw." + prefix, "r")

    fo_train = open(prefix +  ".train", "w")
    fo_test = open(prefix + ".test", "w")
    fo_gen = open(prefix + ".gen", "w")

    fo_train_POS = open(prefix +  ".train.pos", "w")
    fo_test_POS = open(prefix + ".test.pos", "w")
    fo_gen_POS = open(prefix + ".gen.pos", "w")

    count_train = 0
    count_test = 0
    count_gen = 0

    # initialize variables
    train_size = 40000
    test_size = 10000
    gen_size = 10000

    # process raw line by line
    for i, line in enumerate(fi):

        # if done with writing files, break
        if count_train >= train_size and count_test >= test_size and count_gen >= gen_size:
            break

        # if sentence seen, continue
        line = line.strip()
        if line in used_dict:
            continue

        # remember seen
        used_dict[line] = 1

        # tokenize into (word, POS)
        tokens = line.split(' ')[:-1]
        tokens = [(token.split('/')[0], token.split('/')[1]) for token in tokens]
        words = [token[0] for token in tokens]
        POSs = [token[1] for token in tokens]

        # find index of MN
        MNP_idx = POSs.index("MN")

        # determine if sentence is Det [JJ] * MN or Det [adjn, poss] [JJ]* MN
        NP_idxs = [i for i, POS in enumerate(POSs) if POS == "N"]
        sentence_state = get_sentence_state(NP_idxs, MNP_idx, prefix)

        # increment MNP_idx by number of 's occuring
        count_apost = POSs.count("Poss")
        MNP_idx += count_apost

        # join words and POSs
        result_words = " ".join(words) + "\t" + str(MNP_idx) + "\n"
        result_POSs = " ".join(POSs) + "\t" + str(MNP_idx) + "\n"

        # write to corresponding file
        if sentence_state == "Intervening":
            if count_gen >= gen_size:
                continue
            fo_gen.write(result_words)
            fo_gen_POS.write(result_POSs)
            count_gen += 1
        else:
            coin = np.random.choice([0,1], size=1, p=[0.5,0.5])
            if coin:
                if count_test >= test_size:
                    continue
                fo_test.write(result_words)
                fo_test_POS.write(result_POSs)
                count_test += 1
            else:
                if count_train >= train_size:
                    continue
                fo_train.write(result_words)
                fo_train_POS.write(result_POSs)
                count_train += 1

    # print length of files written
    print(f"train: {count_train}")
    print(f"test: {count_test}")
    print(f"gen: {count_gen}")

    fo_train.close()
    fo_test.close()
    fo_gen.close()
    fo_train_POS.close()
    fo_test_POS.close()
    fo_gen_POS.close()

# else if generating for sbjn expt:
elif prefix == "sbjn-join":

    used_dict = {}
    fi = open("raw." + "sbjn", "r")

    fo_train = open(prefix +  ".train", "w")
    fo_test = open(prefix + ".test", "w")
    fo_gen_apost = open(prefix + ".gen_apost", "w")
    fo_gen_comp = open(prefix + ".gen_comp", "w")

    fo_train_POS = open(prefix +  ".train.pos", "w")
    fo_test_POS = open(prefix + ".test.pos", "w")
    fo_gen_apost_POS = open(prefix + ".gen_apost.pos", "w")
    fo_gen_comp_POS = open(prefix + ".gen_comp.pos", "w")

    count_train = 0
    count_test = 0
    count_gen_comp = 0
    count_gen_apost = 0

    # initialize variables
    train_size = 40000
    test_size = 10000
    gen_apost_size = 10000
    gen_comp_size = 10000

    tokenizers = [tokenization.BertTokenizer.from_pretrained(MODEL_ABBREV[bert_model], do_lower_case=True) for bert_model in ["bbu", "blu", "bbmu"]]

    # process raw line by line
    for j, line in tqdm(enumerate(fi)):

        # if done with writing files, break
        if count_train >= train_size and count_test >= test_size and count_gen_apost >= gen_apost_size and count_gen_comp >= gen_comp_size:
            break

        # if sentence seen, continue
        line = line.strip()
        if line in used_dict:
            continue

        # remember seen
        used_dict[line] = 1

        # tokenize into (word, POS)
        tokens = line.split(' ')[:-1]
        tokens = [(token.split('/')[0], token.split('/')[1]) for token in tokens]
        words = [token[0] for token in tokens]
        POSs = [token[1] for token in tokens]

        # find index of MN
        MNP_idx = POSs.index("MN")

        # determine if sentence is Det [JJ] * MN or Det [adjn, poss] [JJ]* MN
        NP_idxs = [i for i, POS in enumerate(POSs) if POS == "N"]
        sentence_state = get_sentence_state(NP_idxs, MNP_idx, "sbjn")

        if "Poss" in POSs and sentence_state == "Intervening":
            sentence_state = "Intervening_apost"
        elif "Poss" not in POSs and sentence_state == "Intervening":
            sentence_state = "Intervening_comp"

        # join apostrophe
        new_words = []
        for i, word in enumerate(words):
            if i == len(words) - 1:
                new_words.append(word)
            elif POSs[i] != "Poss" and POSs[i+1] != "Poss":
                new_words.append(word)
            else:
                if POSs[i+1] == "Poss":
                    continue
                elif POSs[i] == "Poss":
                    new_words.append("".join([words[i-1], words[i]]))
        MNP_idx -= (len(words) - len(new_words))
        words = new_words
        POSs = [POS for POS in POSs if POS != "Poss"]

        assert(len(words) == len(POSs))
        start_MNP_idx = MNP_idx
        end_MNP_idx = MNP_idx + 1
        MNP_test_idx = []
        for i, bert_model in enumerate(["bbu", "blu", "bbmu"]):
            up_to_start = tokenizers[i].tokenize(" ".join(words[:start_MNP_idx]))
            up_to_end = tokenizers[i].tokenize(" ".join(words[:end_MNP_idx]))
            MNP_idx = list(range(len(up_to_start), len(up_to_end)))
            MNP_idx = MNP_idx[0]
            MNP_test_idx.append(MNP_idx)
        assert(all(x == MNP_test_idx[0] for x in MNP_test_idx))

        # print(words)
        # print(POSs)
        # print(tokenizer.tokenize(" ".join(words)))
        # print(MNP_idx)

        # if j == 5:
        #     sys.exit(0)
        MNP_idx = MNP_test_idx[0]

        # join words and POSs
        result_words = " ".join(words) + "\t" + str(MNP_idx) + "\n"
        result_POSs = " ".join(POSs) + "\t" + str(MNP_idx) + "\n"
        # write to corresponding file
        if sentence_state == "Intervening_apost":
            if count_gen_apost >= gen_apost_size:
                continue
            fo_gen_apost.write(result_words)
            fo_gen_apost_POS.write(result_POSs)
            count_gen_apost += 1
        elif sentence_state == "Intervening_comp":
            if count_gen_comp >= gen_comp_size:
                continue
            fo_gen_comp.write(result_words)
            fo_gen_comp_POS.write(result_POSs)
            count_gen_comp += 1
        else:
            coin = np.random.choice([0,1], size=1, p=[0.5,0.5])
            if coin:
                if count_test >= test_size:
                    continue
                fo_test.write(result_words)
                fo_test_POS.write(result_POSs)
                count_test += 1
            else:
                if count_train >= train_size:
                    continue
                fo_train.write(result_words)
                fo_train_POS.write(result_POSs)
                count_train += 1

    # print length of files written
    print(f"train: {count_train}")
    print(f"test: {count_test}")
    print(f"gen_comp: {count_gen_comp}")
    print(f"gen_apost: {count_gen_apost}")

    fo_train.close()
    fo_test.close()
    fo_gen_comp.close()
    fo_gen_apost.close()
    fo_train_POS.close()
    fo_test_POS.close()
    fo_gen_comp_POS.close()
    fo_gen_apost_POS.close()

# else if generating for refl expt:
elif prefix == "refl":

    used_dict = {}
    fi = open("raw." + prefix, "r")

    # initialize variables
    n_types = 9
    files = open_files(prefix, n_types, pos=False)
    files_POS = open_files(prefix, n_types, pos=True)
    counts = [0] * n_types
    sizes = [10000] * n_types

    # process raw line by line
    for i, line in enumerate(fi):

        # if done with writing files, break
        if all(counts[i] >= sizes[i] for i in range(n_types)):
            break

        # if sentence seen, continue
        line = line.strip()
        if line in used_dict:
            continue

        # remember seen
        used_dict[line] = 1

        # tokenize into (word, POS)
        tokens = line.split(' ')[:-1]
        tokens = [(token.split('/')[0], token.split('/')[1]) for token in tokens]
        words = [token[0] for token in tokens]
        POSs = [token[1] for token in tokens]

        # continue if no reflexive
        if "ANT" not in POSs:
            continue

        # determine index of antecedent
        ant_idx = (POSs.index("ANT")-2, POSs.index("ANT")-1)
        del words[POSs.index("ANT")]
        del POSs[POSs.index("ANT")]

        # determine case of reflexive, continuee if no reflexive
        if "Refl_M" in POSs:
            refl_idx = POSs.index("Refl_M")
            refl_gend = "M"
        elif "Refl_F" in POSs:
            refl_idx = POSs.index("Refl_F")
            refl_gend = "F"
        else:
            continue

        # determine idx of NP_o
        NP_idx = [(i-1, i) for i, POS in enumerate(POSs) if POS == "N_F" or POS == "N_M"]
        NP_idx = [pair for pair in NP_idx if pair[0] < refl_idx and pair[0] != ant_idx[0]]

        # NP_idx should be  0,1,2
        assert(len(NP_idx) == 0 or len(NP_idx) == 1 or len(NP_idx) == 2)

        # determine state of sentence
        if "Rel" not in POSs:
            if len(NP_idx) == 0:                    # (0) no Rel, no NP_o
                sentence_state = 0
            elif len(NP_idx) == 1:                  # no Rel, NP_o
                NP_o = NP_idx[0]
                NP_o_gend = "M" if POSs[NP_o[1]] == "N_M" else "F"
                if NP_o_gend == refl_gend:          # (1) no Rel, NP_o match
                    sentence_state = 1
                else:                               # (2) no Rel, NP_o mismatch
                    sentence_state = 2
        else:
            if len(NP_idx) == 0:                    # (0) no Rel, no NP_o
                sentence_state = 0
            elif len(NP_idx) == 1:                  # Rel, no NP_o, NP_r
                NP_r = NP_idx[0]
                NP_r_gend = "M" if POSs[NP_r[1]] == "N_M" else "F"
                if NP_r_gend == refl_gend:          # (3) Rel, no NP_o, NP_r match
                    sentence_state = 3
                else:                               # (4) Rel, no NP_o, NP_r mismatch
                    sentence_state = 4
            elif len(NP_idx) == 2:                  # Rel, NP_o, NP_r
                NP_r = NP_idx[0]
                NP_r_gend = "M" if POSs[NP_r[1]] == "N_M" else "F"
                NP_o = NP_idx[1]
                NP_o_gend = "M" if POSs[NP_o[1]] == "N_M" else "F"
                if NP_o_gend == refl_gend:          # Rel, NP_o match, NP_r
                    if NP_r_gend == refl_gend:      # (5) Rel, NP_o match, NP_r match
                        sentence_state = 5
                    else:                           # (6) Rel, NP_o match, NP_r mismatch
                        sentence_state = 6
                else:                               # Rel, NP_o mismatch, NP_r
                    if NP_r_gend == refl_gend:      # (7) Rel, NP_o mismatch, NP_r match
                        sentence_state = 7
                    else:
                        sentence_state = 8          # (8) Rel, NP_o mismatch, NP_r mismatch

        # join words and POSs
        result_words = " ".join(words)
        result_words += "\t" + str(refl_idx)
        result_words += "\t" + str(ant_idx[0]) + " " + str(ant_idx[1]) + "\t"
        for pair in NP_idx:
            result_words += str(pair[0]) + " " + str(pair[1]) + " "
        result_words += "\n"

        result_POSs = " ".join(POSs)
        result_POSs += "\t" + str(refl_idx)
        result_POSs += "\t" + str(ant_idx[0]) + " " + str(ant_idx[1]) + "\t"
        for pair in NP_idx:
            result_POSs += str(pair[0]) + " " + str(pair[1]) + " "
        result_POSs += "\n"

        # write to corresponding file
        if counts[sentence_state] >= sizes[sentence_state]:
            continue
        files[sentence_state].write(result_words)
        files_POS[sentence_state].write(result_POSs)
        counts[sentence_state] += 1

    # print length of files written
    for i, file in enumerate(files):
        print(f"{i}: {counts[i]}")

    close_files(files)

# else if generating for agr expt:
elif prefix == "agr":

    used_dict = {}
    fi = open("raw." + prefix, "r")

    # initialize variables
    n_types = 20
    files = open_files(prefix, n_types, pos=False)
    files_POS = open_files(prefix, n_types, pos=True)
    counts = [0] * n_types
    sizes = [10000] * n_types

    # process raw line by line
    for i, line in enumerate(fi):

        # if done with writing files, break
        if all(counts[i] >= sizes[i] for i in range(n_types)):
            break

        # if sentence seen, continue
        line = line.strip()
        if line in used_dict:
            continue

        # remember seen
        used_dict[line] = 1

        # tokenize into (word, POS)
        tokens = line.split(' ')[:-1]
        tokens = [(token.split('/')[0], token.split('/')[1]) for token in tokens]
        words = [token[0] for token in tokens]
        POSs = [token[1] for token in tokens]

        # determine index of sbjn
        sbjn_idx = (POSs.index("AGR")-2, POSs.index("AGR")-1)
        del words[POSs.index("AGR")]
        del POSs[POSs.index("AGR")]

        # determine index of AAux
        aaux_idx = POSs.index("AAux")-1
        del words[POSs.index("AAux")]
        del POSs[POSs.index("AAux")]

        # determine num of NP_s
        sbjn_num = "sg" if POSs[sbjn_idx[1]] == "N_sg" else "pl"

        # determine idx of NP_o
        NP_idx = [(i-1, i) for i, POS in enumerate(POSs) if POS == "N_sg" or POS == "N_pl"]
        NP_idx = [pair for pair in NP_idx if pair[0] < aaux_idx and pair[0] != sbjn_idx[0]]

        # NP_idx should be  0,1,2
        assert(len(NP_idx) == 0 or len(NP_idx) == 1)

        # determine state of sentence
        if "Prep" not in POSs and "Rel" not in POSs:
            assert(len(NP_idx) == 0)
            if sbjn_num == "sg":
                if POSs[aaux_idx] == "Aux":         # (0) norm, sbjn s, m
                    sentence_state = 0
                else:                               # (1) norm, sbjn s, nm
                    sentence_state  = 1
            else:
                if POSs[aaux_idx] == "Aux":         # (2) norm, sbjn p, m
                    sentence_state = 2
                else:                               # (3) norm, sbjn p, nm
                    sentence_state  = 3
        elif "Prep" in POSs:
            assert(len(NP_idx) == 1)
            NP_p = NP_idx[0]
            NP_p_num = "sg" if POSs[NP_p[1]] == "N_sg" else "pl"
            if sbjn_num == "sg":
                if NP_p_num == "sg":
                    if POSs[aaux_idx] == "Aux":     # (4) prep, sbjn s, NP_p s, m
                        sentence_state = 4
                    else:                           # (5) prep, sbjn s, NP_p s, nm
                        sentence_state  = 5
                else:
                    if POSs[aaux_idx] == "Aux":     # (6) prep, sbjn s, NP_p p, m
                        sentence_state = 6
                    else:                           # (7) prep, sbjn s, NP_p p, nm
                        sentence_state  = 7
            else:
                if NP_p_num == "sg":
                    if POSs[aaux_idx] == "Aux":     # (8) prep, sbjn p, NP_p s, m
                        sentence_state = 8
                    else:                           # (9) prep, sbjn p, NP_p s, nm
                        sentence_state  = 9
                else:
                    if POSs[aaux_idx] == "Aux":     # (10) prep, sbjn p, NP_p p, m
                        sentence_state = 10
                    else:                           # (11) prep, sbjn p, NP_p p, nm
                        sentence_state  = 11
        elif "Rel" in POSs:
            assert(len(NP_idx) == 0 or len(NP_idx) == 1)
            if sbjn_num == "sg":
                if len(NP_idx) == 0:
                    if POSs[aaux_idx] == "Aux":         # (0) relc, sbjn s, m
                        sentence_state = 0
                    else:                               # (1) relc, sbjn s, nm
                        sentence_state  = 1
                else:
                    NP_r = NP_idx[0]
                    NP_r_num = "sg" if POSs[NP_r[1]] == "N_sg" else "pl"
                    if NP_r_num == "sg":
                        if POSs[aaux_idx] == "Aux":     # (12) relc, sbjn s, NP_r s, m
                            sentence_state = 12
                        else:                           # (13) relc, sbjn s, NP_r s, nm
                            sentence_state  = 13
                    else:
                        if POSs[aaux_idx] == "Aux":     # (14) relc, sbjn s, NP_r p, m
                            sentence_state = 14
                        else:                           # (15) relc, sbjn s, NP_r p, nm
                            sentence_state  = 15
            elif sbjn_num == "pl":
                if len(NP_idx) == 0:
                    if POSs[aaux_idx] == "Aux":         # (2) relc, sbjn p, m
                        sentence_state = 2
                    else:                               # (3) relc, sbjn p, nm
                        sentence_state  = 3
                else:
                    NP_r = NP_idx[0]
                    NP_r_num = "sg" if POSs[NP_r[1]] == "N_sg" else "pl"
                    if NP_r_num == "sg":
                        if POSs[aaux_idx] == "Aux":     # (16) relc, sbjn p, NP_r s, m
                            sentence_state = 16
                        else:                           # (17) relc, sbjn p, NP_r s, nm
                            sentence_state  = 17
                    else:
                        if POSs[aaux_idx] == "Aux":     # (18) relc, sbjn p, NP_r p, m
                            sentence_state = 18
                        else:                           # (19) relc, sbjn p, NP_r p, nm
                            sentence_state  = 19

        # join words and POSs
        result_words = " ".join(words)
        result_words += "\t" + str(aaux_idx)
        result_words += "\t" + str(sbjn_idx[0]) + " " + str(sbjn_idx[1]) + "\t"
        for pair in NP_idx:
            result_words += str(pair[0]) + " " + str(pair[1]) + " "
        result_words += "\n"

        result_POSs = " ".join(POSs)
        result_POSs += "\t" + str(aaux_idx)
        result_POSs += "\t" + str(sbjn_idx[0]) + " " + str(sbjn_idx[1]) + "\t"
        for pair in NP_idx:
            result_POSs += str(pair[0]) + " " + str(pair[1]) + " "
        result_POSs += "\n"

        # write to corresponding file
        if counts[sentence_state] >= sizes[sentence_state]:
            continue
        files[sentence_state].write(result_words)
        files_POS[sentence_state].write(result_POSs)
        counts[sentence_state] += 1

    # print length of files written
    for i, file in enumerate(files):
        print(f"{i}: {counts[i]}")

    close_files(files)

# else if generating for dill1 expt: note, process dill1 then dill2
elif prefix == "dill":

    for bert_model in ["bbmu", "blu", "bbmu"]:

        # init tokenizer
        tokenizer = tokenization.BertTokenizer.from_pretrained(MODEL_ABBREV[bert_model], do_lower_case=True)

        # initialize variables
        n_types = 6
        files = open_files(f"{prefix}.{bert_model}", n_types, pos=False)
        counts = [0] * n_types

        used_dict = {}
        fi = open("raw." + prefix + "1", "r")

        # process raw line by line
        for i, line in enumerate(fi):

            # if sentence seen, continue
            line = line.strip()
            if line in used_dict:
                continue

            # if sentence is type 3,4,7,8, ungrammatical, continue
            if line[0] == str(3) or line[0] == str(4) or line[0]  == str(7) or line[0] == str(8):
                continue

            # remember seen
            used_dict[line] = 1

            # determine sentence state
            if line[0] == str(1):
                sentence_state = 0
            elif line[0] == str(2):
                sentence_state = 1
            elif line[0] == str(5):
                sentence_state = 2
            elif line[0] == str(6):
                sentence_state = 3

            # tokenize into words
            tokens = line.split(' ')
            tokens[:-1] = tokens[:-1][:-1] # remove period from last token
            tokens = tokens[2:] # remove first two numbers
            words = tokens

            # determine aaux_idx and sbjn_idx and NP_r idx
            if sentence_state == 0 or sentence_state == 1:

                start_aaux_idx = 10
                end_aaux_idx = 11
                up_to_start = tokenizer.tokenize(" ".join(words[:start_aaux_idx]))
                up_to_end = tokenizer.tokenize(" ".join(words[:end_aaux_idx]))
                aaux_idx = list(range(len(up_to_start), len(up_to_end)))

                start_sbjn_idx = 0
                end_sbjn_idx = 3
                up_to_start = tokenizer.tokenize(" ".join(words[:start_sbjn_idx]))
                up_to_end = tokenizer.tokenize(" ".join(words[:end_sbjn_idx]))
                sbjn_idx = list(range(len(up_to_start), len(up_to_end)))

                start_NP_r_idx = 6
                end_NP_r_idx = 9
                up_to_start = tokenizer.tokenize(" ".join(words[:start_NP_r_idx]))
                up_to_end = tokenizer.tokenize(" ".join(words[:end_NP_r_idx]))
                NP_r_idx = list(range(len(up_to_start), len(up_to_end)))

                # join words and POSs
                result_words = " ".join(words)
                result_words += "\t" + str(aaux_idx[0])
                result_words += "\t"
                for num in sbjn_idx:
                    result_words += str(num) + " "
                result_words += "\t"
                for num in NP_r_idx:
                    result_words += str(num) + " "
                result_words += "\n"


            elif sentence_state == 2 or sentence_state == 3:

                start_refl_idx = 11
                end_refl_idx = 12
                up_to_start = tokenizer.tokenize(" ".join(words[:start_refl_idx]))
                up_to_end = tokenizer.tokenize(" ".join(words[:end_refl_idx]))
                refl_idx = list(range(len(up_to_start), len(up_to_end)))

                start_ant_idx = 0
                end_ant_idx = 3
                up_to_start = tokenizer.tokenize(" ".join(words[:start_ant_idx]))
                up_to_end = tokenizer.tokenize(" ".join(words[:end_ant_idx]))
                ant_idx = list(range(len(up_to_start), len(up_to_end)))

                start_NP_r_idx = 6
                end_NP_r_idx = 9
                up_to_start = tokenizer.tokenize(" ".join(words[:start_NP_r_idx]))
                up_to_end = tokenizer.tokenize(" ".join(words[:end_NP_r_idx]))
                NP_r_idx = list(range(len(up_to_start), len(up_to_end)))

                # join words and POSs
                result_words = " ".join(words)
                result_words += "\t" + str(refl_idx[0])
                result_words += "\t"
                for num in ant_idx:
                    result_words += str(num) + " "
                result_words += "\t"
                for num in NP_r_idx:
                    result_words += str(num) + " "
                result_words += "\n"

            # write to corresponding file
            files[sentence_state].write(result_words)
            counts[sentence_state] += 1

        fi.close()
        fi = open("raw." + prefix + "2", "r")

        # process raw line by line
        for i, line in enumerate(fi):

            # if sentence seen, continue
            line = line.strip()
            if line in used_dict:
                continue

            # if sentence is type 3,4,7,8, ungrammatical, continue
            if line[0] == str(3) or line[0] == str(4) or line[0]  == str(5) or line[0] == str(6):
                continue

            # remember seen
            used_dict[line] = 1

            # determine sentence state
            if line[0] == str(1):
                sentence_state = 2
            elif line[0] == str(2):
                sentence_state = 3
            elif line[0] == str(7):
                sentence_state = 4
            elif line[0] == str(8):
                sentence_state = 5

            # tokenize into words
            tokens = line.split(' ')
            tokens[:-1] = tokens[:-1][:-1] # remove period from last token
            tokens = tokens[2:] # remove first two numbers
            words = tokens

            start_refl_idx = 11
            end_refl_idx = 12
            up_to_start = tokenizer.tokenize(" ".join(words[:start_refl_idx]))
            up_to_end = tokenizer.tokenize(" ".join(words[:end_refl_idx]))
            refl_idx = list(range(len(up_to_start), len(up_to_end)))

            start_ant_idx = 0
            end_ant_idx = 3
            up_to_start = tokenizer.tokenize(" ".join(words[:start_ant_idx]))
            up_to_end = tokenizer.tokenize(" ".join(words[:end_ant_idx]))
            ant_idx = list(range(len(up_to_start), len(up_to_end)))

            start_NP_r_idx = 6
            end_NP_r_idx = 9
            up_to_start = tokenizer.tokenize(" ".join(words[:start_NP_r_idx]))
            up_to_end = tokenizer.tokenize(" ".join(words[:end_NP_r_idx]))
            NP_r_idx = list(range(len(up_to_start), len(up_to_end)))

            # join words and POSs
            result_words = " ".join(words)
            result_words += "\t" + str(refl_idx[0])
            result_words += "\t"
            for num in ant_idx:
                result_words += str(num) + " "
            result_words += "\t"
            for num in NP_r_idx:
                result_words += str(num) + " "
            result_words += "\n"

            # write to corresponding file
            files[sentence_state].write(result_words)
            counts[sentence_state] += 1

        # print length of files written
        print(bert_model)
        for i, file in enumerate(files):
            print(f"{i}: {counts[i]}")
        print()

        fi.close()
        close_files(files)

# close files
fi.close()
