import argparse
import csv
import operator
import random

import nltk
import numpy as np
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='paws/final/test.tsv')
    parser.add_argument('--output', type=str, default='paws.tsv')
    args = parser.parse_args()
    in_file = args.input
    out_file = args.output

    w2v = KeyedVectors.load_word2vec_format("../w2v/wiki.multi.en.vec")

    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download('punkt')

    pairs = {}
    data = []

    with open(in_file, "r") as in_:
        tsv = csv.reader(in_, delimiter="\t")
        next(tsv)
        for line in tsv:
            id_, sent1, sent2, label = line  #
            data.append((sent1, sent2, int(label)))

    sentences = list(set([s1 for s1, s2, l in data] + [s2 for s1, s2, l in data]))

    used_sents = []
    sents = []
    sents0 = []
    sents1 = []
    lo_diff = []
    for sentence in sentences:
        if sentence in used_sents:
            continue
        label0_candidates = [sent2 for sent1, sent2, label in data if
                             sent1 == sentence and sent2 not in used_sents and label == 0]
        label1_candidates = [sent1 for sent1, sent2, label in data if
                             sent2 == sentence and sent1 not in used_sents and label == 1]

        if len(label0_candidates) == 0 or len(label1_candidates) == 0:
            continue

        bleu_label0_scores = [(s, sentence_bleu([word_tokenize(sentence)], word_tokenize(s), (1, 0))) for s in
                              label0_candidates]
        bleu_label1_scores = [(s, sentence_bleu([word_tokenize(sentence)], word_tokenize(s), (1, 0))) for s in
                              label1_candidates]

        bleu_label0_scores = sorted(bleu_label0_scores, key=operator.itemgetter(1))
        bleu_label1_scores = sorted(bleu_label1_scores, key=operator.itemgetter(1))

        label0_sentence = bleu_label0_scores[-1][0]
        label1_sentence = bleu_label1_scores[0][0]
        label0_score = bleu_label0_scores[-1][1]
        label1_score = bleu_label1_scores[0][1]

        if label1_score < label0_score:
            sents0.append(label0_sentence)
            sents1.append(label1_sentence)
            sents.append(sentence)
            lo_diff.append(label0_score - label1_score)

            used_sents.append(label0_sentence)
            used_sents.append(label1_sentence)

            used_sents.append(sentence)

    lowest_lo = list(reversed(sorted(list(enumerate(lo_diff)), key=lambda t: t[1])))

    new_dataset = []
    for ind, _ in lowest_lo[:300]:
        new_dataset.append((0, sents[ind], sents0[ind]))
        new_dataset.append((1, sents[ind], sents1[ind]))

    with open(out_file, "w") as out:
        tsv_writer = csv.writer(out, delimiter="\t")
        tsv_writer.writerows(new_dataset)
