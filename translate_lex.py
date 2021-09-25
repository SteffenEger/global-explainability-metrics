from __future__ import absolute_import, division, unicode_literals

import argparse
import csv
import os
import pickle
import string
import time

import httpcore
import jieba
from googletrans import Translator
from nltk.tokenize import word_tokenize



def translate(input, lang, output):
    translator = Translator(timeout=None)
    if lang == "zh":
        lang = "zh-cn"

    translated_sentence = {}
    translated_words = {}

    if os.path.exists(output):
        with open(output, "rb") as out_file:
            final_dict = pickle.load(out_file)
            translated_sentence = final_dict["sentence"]
            translated_words = final_dict["word"]
            translated_sentence = translated_sentence
            translated_words = translated_words

    try:
        with open(input, encoding='utf-8', errors="surrogateescape") as tsvfile:
            reader = list(csv.reader(tsvfile, delimiter='\t'))
            for row in reader:
                sentence = row[2]
                if sentence in translated_words and sentence in translated_sentence:
                    continue
                sent_temp_word = []
                reference = row[2].replace('.', '. ').replace("//"," ").replace("/"," ")

                print(reference)
                if lang == "zh-cn":
                    reference_tokenized = list(jieba.cut(reference))
                else:
                    reference_tokenized = word_tokenize(reference, language="german")
                for word in reference_tokenized:
                    if word.isspace():
                        continue
                    print("Word", word)
                    sent_temp_word.extend(word_tokenize(translator.translate(word.replace('.', '. '),  dest='en').text))
                    time.sleep(1.0)
                translated_words[sentence] = sent_temp_word
                sent_temp_sentence = word_tokenize(translator.translate(reference, dest='en', src=lang).text)
                translated_sentence[sentence] = sent_temp_sentence
                print(" ".join(sent_temp_word))
                print(sent_temp_sentence)
                time.sleep(1.0)
            all_done = True
    except httpcore._exceptions.ConnectError:
        print("ConnectError")
        all_done = False

    if not os.path.exists(os.path.dirname(output)):
        os.mkdir(os.path.dirname(output))
    with open(output, "wb") as out_file:
        final_dict = {"word": translated_words, "sentence": translated_sentence}
        pickle.dump(final_dict, out_file)
    return all_done

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='./data/raw/wmt_de_full.tsv')
parser.add_argument('--lang', type=str, default='de')
parser.add_argument('--output', type=str, default='data/translated/wmt_de.pkl')
args = parser.parse_args()


all_done = False
tries = 0
while not all_done and tries < 50:
    all_done = translate(args.input, args.lang, args.output)
    tries += 1
print("Done?", all_done)

