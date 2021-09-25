import argparse
import csv

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/raw/paws.tsv')
args = parser.parse_args()
in_file = args.input
with open(in_file, "r") as inf:
    lex_0 = 0
    lex_1 = 1
    lines = list(csv.reader(inf, delimiter="\t"))

    last_ov = None

    for line in lines:
        label, sent1, sent2 = line
        sent2 = sent2.replace(" .", ".").replace(" \'s", "\'s").replace(" \'m", "\'m").replace("\"", "").replace("`","").replace("\'", "")
        sent1 = sent1.replace(" .", ".").replace(" \'s", "\'s").replace(" \'m", "\'m").replace("\"", "").replace("`","").replace("\'", "")
        ov = sentence_bleu([word_tokenize(sent1)], word_tokenize(sent2), (1., 0.))

        if int(label) == 0:
            lex_0 += ov
            last_ov = ov
        else:
            lex_1 += ov

r0 = lex_0 / (len(lines) / 2)
r1 = lex_1 / (len(lines) / 2)

print("A,C",r0)
print("A,B",r1)
