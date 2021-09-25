#from __future__ import absolute_import, division, unicode_literals

import csv
from googletrans import Translator
import time
import httpcore
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, default='./data/raw/sts-train.csv')
parser.add_argument('--lang', type=str, default='zh')
parser.add_argument('--output_file', type=str, default='translate_output.tsv')
args = parser.parse_args()


# translate one sentence from each sentence pair into target language
with open(args.input_file, encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    originals = []
    translated = []
    sem_scores = []
    translator = Translator(timeout=None)
    i = 0
    for row in reader:
            row = [row[4], row[5], row[6]]
            if args.lang == "zh":
                args.lang = "zh-cn"
            try:
                trans = translator.translate(row[2], src='en', dest=args.lang)
                translated.append(trans.text)
            except httpcore._exceptions.ReadTimeout:
                print("translator timeout, printing interim results to output file")
                with open(args.output_file, 'wt', encoding='utf-8', newline='') as output:
                    tsv_writer = csv.writer(output, delimiter='\t')
                    for i in range(len(translated)):
                        tsv_writer.writerow([sem_scores[i], originals[i], translated[i]])
                break
            originals.append(row[1])

            sem_scores.append(round(float(row[0]) / 5, 4))
            # without wait time, the translate service can return untranslated sentences
            time.sleep(1)
            i += 1
with open(args.output_file, 'wt', encoding='utf-8', newline='') as output:
    tsv_writer = csv.writer(output, delimiter='\t')
    for i in range(len(translated)):
        tsv_writer.writerow([sem_scores[i], originals[i], translated[i]])
