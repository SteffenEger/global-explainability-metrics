from __future__ import absolute_import, division, unicode_literals
import fasttext
import csv
import stanza
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='./data/morph/wmt_plaintext_en.tsv')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--output', type=str, default='embed_output.tsv')
args = parser.parse_args()

stanza.download('de', processors='tokenize', package='gsd')
stanza.download('en', processors='tokenize', package='gum')
en_pipe = stanza.Pipeline('en',  processors='tokenize', package='gum')
de_pipe = stanza.Pipeline('de',  processors='tokenize', package='gsd')

# train a Fasttext model on the plaintext input file (with one sentence per line)
model = fasttext.train_unsupervised(args.input, epoch=10, lr=0.2)


all_words = []
de_sent = []
en_sent = []

# read out all single words from the input file
with open(args.input, encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    i = 0
    for row in reader:
        if i % 2 == 1:
            all_words.extend([word for sen in en_pipe(row[0]).sentences for word in sen.words])
        else:
            if args.lang == "de":
                all_words.extend([word for sen in de_pipe(row[0]).sentences for word in sen.words])
            else:
                all_words.extend([word for sen in en_pipe(row[0]).sentences for word in sen.words])
        i += 1

all_words = set([word.text for word in all_words])

# save the embedding for each word
with open(args.output, 'wt', encoding='utf-8', newline='') as output:
    tsv_writer = csv.writer(output, delimiter=' ')
    i = 0
    for word in all_words:
        out = [word]
        out.extend([str(j) for j in model.get_word_vector(word)])
        tsv_writer.writerow(out)
        i += 1
