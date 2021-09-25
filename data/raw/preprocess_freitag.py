import argparse
import csv
import operator
import random

import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

random.seed(42)

if __name__ == '__main__':
    nltk.download("averaged_perceptron_tagger")

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='freitag/newstest2019-ende-src.en')
    parser.add_argument('--ref', type=str, default='freitag/wmt19-ende-arp.ref.gt')
    parser.add_argument('--output', type=str, default='freitag.tsv')
    args = parser.parse_args()
    in_file_orig = args.source
    in_file_trans = args.ref
    out_file = args.output

    orig_sents = []
    trans_sents = []

    with open(in_file_orig, "r") as lines:
        for line in lines:
            orig_sents.append(line.strip())

    with open(in_file_trans, "r") as lines:
        for line in lines:
            trans_sents.append(line.strip())

    new_dataset = []

    for i in range(len(trans_sents)):

        orig_tokens = word_tokenize(orig_sents[i])

        annotated_tokens = nltk.pos_tag(orig_tokens)

        noun_indices = [k for k, (word, pos) in enumerate(annotated_tokens) if pos == "NN"]


        if len(noun_indices) < 2:
            continue

        while noun_indices == sorted(noun_indices):
            random.shuffle(noun_indices)

        new_sent = []
        k = 0
        for j in range(len(orig_tokens)):
            if j in noun_indices:
                new_sent.append(orig_tokens[noun_indices[k]])
                k += 1
            else:
                new_sent.append(orig_tokens[j])

        shuffled_sent = " ".join(new_sent)
        new_dataset.append((0, orig_sents[i], shuffled_sent))
        new_dataset.append((1, orig_sents[i], trans_sents[i]))

    with open(out_file, "w") as out:
        tsv_writer = csv.writer(out, delimiter="\t")
        tsv_writer.writerows(new_dataset)
