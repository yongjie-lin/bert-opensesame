import random
import sys

train_size = 40000
test_size = 10000
gen_size = 10000

prefix = sys.argv[1]

fi = open("raw." + prefix, "r")

fo_train = open(prefix +  ".train", "w")
fo_test_basic = open(prefix + ".test", "w")
fo_gen = open(prefix + ".gen", "w")

fo_train_POS = open(prefix +  ".train.pos", "w")
fo_test_basic_POS = open(prefix + ".test.pos", "w")
fo_gen_POS = open(prefix + ".gen.pos", "w")

used_dict = {}

count_train = 0
count_basic = 0
count_gen = 0

delList = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8",
        "VI1", "VI2", "VI3", "VT1", "VT2", "VT3", "VT4", "VT5", "VI", "VT", "Det", "Rel", "Prep", "JJ"]
delDict = {}
for item in delList:
    delDict[item] = 1


def questionify(sent):
    sent[-2] = "?"

    after = sent[-2:]
    sent = sent[:-2]

    words = [sent[i] for i in range(1, len(sent), 2)]
    POSs = [sent[i] for i in range(0, len(sent), 2)]

    if "AUX4" in POSs:
        ind = POSs.index("AUX4")
    else:
        ind = POSs.index("AUX5")

    if words[ind] == "does":
        words[ind + 1] = words[ind + 1][:-1]
        POSs[ind + 1] = POSs[ind + 1][:-1]

    newWords = [words[ind]] + words[:ind] + words[ind + 1:]
    newPOSs = [POSs[ind]] + POSs[:ind] + POSs[ind + 1:]
    newSent = [val for pair in zip(newWords, newPOSs) for val in pair] + after

    return newSent

def process(sent):
    if sent[-1] == "quest":
        quest = 1
    else:
        quest = 0

    newSent = []
    newSent_POS = []
    for word in sent:
        if word not in delDict:
            newSent.append(word)
        if word in delDict:
            newSent_POS.append(word)

    newNewSent = []
    prevWord = ""
    for word in newSent:
        if prevWord == "does" and word[-1] == "s":
            newNewSent.append(word[:-1])
        else:
            newNewSent.append(word)
        prevWord = word

    return " ".join(newNewSent), " ".join(newSent_POS)

count_orc = 0
count_src = 0
aux_list = ["can", "may", "will", "might", "must", "would", "could", "should", "mayn't", "won't", "mightn't", "wouldn't", "shouldn't", "shan't", "do", "does", "don't", "doesn't"]
aux_dict = {}
for aux in aux_list:
    aux_dict[aux] = 1

def get_auxes(words):
    aux_set = []
    for word in words:
        if word in aux_dict:
            aux_set.append(word)

    new_aux_set = []
    for aux in aux_set:
        if "do" not in aux:
            new_aux_set.append("aux")
        else:
            new_aux_set.append(aux)

    return new_aux_set

for i, line in enumerate(fi):

    if count_train >= train_size and count_basic >= test_size and count_gen >= gen_size:
        break

    sent = line.strip()
    if sent in used_dict:
        continue

    used_dict[sent] = 1

    words = sent.split()

    if words[5] == "that" or words[5] == "who":
        rel_on_subj = 1
    else:
        rel_on_subj = 0

    choose = random.getrandbits(1)

    quest = True
    if count_train >= train_size and count_basic >= test_size:
        quest = True
        choose = 1
    if quest:
        words.append("quest")
    else:
        words.append("decl")

    if quest:
        statement, POS_statement = process(words)
        question, POS_question = process(questionify(words))
        result = statement + "\t" + question + "\n"
        result_POS = POS_statement + "\t" + POS_question + "\n"
    else:
        result = process(words) + "\t" + process(words) + "\n"

    if choose == 0 and count_basic >= test_size:
        choose = 1

    if rel_on_subj and quest and count_gen < gen_size:
        if count_gen < test_size:
            words_auxes = get_auxes(words)
            if words_auxes == ["do", "don't"] or words_auxes == ["don't", "do"] or words_auxes == ["does", "doesn't"] or words_auxes == ["doesn't", "does"] or words_auxes == ["aux", "aux"]:
                if count_src <= 6666:
                    fo_gen.write(result)
                    fo_gen_POS.write(result_POS)
                    count_gen += 1
                    count_src += 1
                else:
                    fo_gen.write(result)
                    fo_gen_POS.write(result_POS)
                    count_gen += 1
                    count_orc += 1
            #else:
            #    print(words_auxes)
    elif choose == 0 and count_basic < test_size and (not rel_on_subj or not quest):
        if not rel_on_subj or not quest:
            fo_test_basic.write(result)
            fo_test_basic_POS.write(result_POS)
            count_basic += 1
    elif count_train < train_size:
        fo_train.write(result)
        fo_train_POS.write(result_POS)
        count_train += 1
    # else:
    #     break

print(count_orc, count_src, count_gen, test_size)
