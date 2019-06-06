import unicodedata
from pytorch_pretrained_bert import BertTokenizer

bert_model = 'bert-base-uncased'
file_type = {'train': 'train', 'dev': 'test', 'test': 'gen'}
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
data_str = '' # 'data_nth/'

MIN_LENGTH = 10
MAX_LENGTH = 30

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

for file_str in ('train', 'dev', 'test'):
    in_str = data_str + file_str + '.tsv'
    out_str = data_str + 'nth.' + file_type[file_str]
    out_str_pos = data_str + 'nth.' + file_type[file_str] + '.pos'

    with open(in_str, 'r') as f_in, open(out_str, 'w') as f_out, open(out_str_pos, 'w') as f_pos:
        lines = [normalize_string(line).split() for line in f_in]
        lines = [line if line else ['\n', '\n'] for line in lines]
        text, pos = (' '.join(s).lstrip() for s in zip(*lines))

        def format_text(s):
            tokenized_lines = (tokenizer.tokenize(line) for line in s.split('\n'))
            tokenized_lines = [' '.join(toks) for toks in tokenized_lines if MIN_LENGTH <= len(toks) <= MAX_LENGTH]
            return '\n'.join(tokenized_lines)

        def format_pos(s):
            return '\n'.join(line.lstrip() for line in s.split('\n'))

        text = format_text(text)
        pos = format_pos(pos)
        f_out.write(text)
        f_pos.write(pos)
