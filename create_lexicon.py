#from __future__ import absolute_import, division, unicode_literals

import csv

import stanza
import random
import argparse

random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='./data/morph/wmt_plaintext_en.tsv')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--output', type=str, default='output_lexicon.tsv')
args = parser.parse_args()


stanza.download('en', processors='tokenize,  pos', package='gum')
stanza.download('de', processors='tokenize, pos', package='gsd')
en_pipe = stanza.Pipeline('en',  processors='tokenize,  pos', package='gum')
de_pipe = stanza.Pipeline('de', processors='tokenize, pos', package='gsd')

with open(args.input, encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    en_words = []
    zh_words = []
    de_words = []

    i = 0
    for row in reader:
        # read out and tag all sentences from the input file, saving all words with their tags
        if i % 2 == 1:
            en_words.extend([word for sen in en_pipe(row[0]).sentences for word in sen.words])
        else:
            if args.lang == "de":
                de_words.extend([word for sen in de_pipe(row[0]).sentences for word in sen.words])
            else:
                en_words.extend([word for sen in en_pipe(row[0]).sentences for word in sen.words])
        i += 1

random.shuffle(en_words)
random.shuffle(de_words)

pairs = []
# for each word, attempt to find a different, matching word (here in a sample from the wordlist)
for i, word in enumerate(en_words):
    print(i)
    # In German, search for English words whose tags are a subset of the German word's tags
    if word.feats and args.lang == "de":
        s = set(word.feats.split('|'))
        for word2 in random.sample(de_words, 5000):
            if word2.feats:
                s2 = set(word2.feats.split('|'))
                s2_new = []
                for feat in s2:
                    if not feat.startswith("Foreign"):
                        s2_new.append(feat)
                s2_new = set(s2_new)
                if s == s2_new and not word.text == word2.text:
                    pairs.append((word.text, word2.text))
                    print(word.text, word2.text, word.feats)
                    break
    # In English search for words that have exactly matching tags
    if word.feats and args.lang == "en":
        for word2 in random.sample(en_words, 5000):
            if word2.feats:
                if set(word.feats.split('|')) == set(word2.feats.split('|')) and not word.text == word2.text:
                    pairs.append((word.text, word2.text))
                    print(word.text, word2.text, word.feats)
                    break


pairs = set(pairs)
# write out results
with open(args.output, 'wt', encoding='utf-8', newline='') as output:
    tsv_writer = csv.writer(output, delimiter='\t')
    for pair in pairs:
        tsv_writer.writerow([' '.join(pair)])
