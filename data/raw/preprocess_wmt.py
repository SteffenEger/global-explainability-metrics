import argparse
import csv
import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='en')
    args = parser.parse_args()
    lang = args.lang
    if lang == "en":
        wmt_files = glob.glob("WMT[0-9][0-9]_testset/*-en.tsv", recursive=True)
    else:
        wmt_files = glob.glob("WMT[0-9][0-9]_testset/*"+lang+"-en.tsv", recursive=True)

    wmt_files = sorted(wmt_files)

    print(wmt_files)

    data = []
    for path in wmt_files:
        with open(path) as file:
            read_tsv = csv.reader(file, delimiter="\t")
            next(read_tsv)
            for row in read_tsv:
                lp, src, translation, ref, human_score = row[:5]
                if lang == "en":
                    data.append((human_score, translation, ref))
                else:
                    data.append((human_score, translation, src))

    with open("wmt_" + lang + "_full.tsv", "w") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerows(data)
