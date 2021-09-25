import argparse
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='../raw/mlqe_zh.tsv')
parser.add_argument('--output', type=str, default='mlqe_plaintext_zh.tsv')
args = parser.parse_args()
input_file = args.input
output_file = args.output

sents = []
with open(input_file) as file:
    read_tsv = csv.reader(file, delimiter="\t")
    for row in read_tsv:
        _, sent1, sent2 = row
        sents.append(sent1)
        sents.append(sent2)

with open(output_file, "w") as out:
    for sent in sents:
        out.write(sent + "\n")
