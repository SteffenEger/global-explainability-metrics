import argparse
import csv
import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='de')
    args = parser.parse_args()
    lang = args.lang

    data = []
    for path in [ "mlqe-pe/test20.en"+lang+".df.short.tsv"]:
        with open(path) as file:
            read_tsv = csv.reader(file, delimiter="\t")
            next(read_tsv)
            for row in read_tsv:
                _, src, translation, _, human_mean_score, _ , z_mean = row[:7]
                data.append((z_mean, src, translation))

    with open("mlqe_" + lang + ".tsv", "w") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerows(data)
