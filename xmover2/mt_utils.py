import glob
from io import StringIO
import pandas as pd
import os
from scipy.stats import pearsonr


def find_corpus(name):
    WMT2017 = dict({
        "newstest2017-csen-ref.en": "cs-en",
        "newstest2017-deen-ref.en": "de-en",
        "newstest2017-fien-ref.en": "fi-en",
        "newstest2017-lven-ref.en": "lv-en",
        "newstest2017-ruen-ref.en": "ru-en",
        "newstest2017-tren-ref.en": "tr-en",
        "newstest2017-zhen-ref.en": "zh-en"
    })

    WMT2018 = dict({
        "newstest2018-csen-ref.en": "cs-en",
        "newstest2018-deen-ref.en": "de-en",
        "newstest2018-eten-ref.en": "et-en",
        "newstest2018-fien-ref.en": "fi-en",
        "newstest2018-ruen-ref.en": "ru-en",
        "newstest2018-tren-ref.en": "tr-en",
        "newstest2018-zhen-ref.en": "zh-en",
    })

    WMT2019 = dict({
        "newstest2019-deen-ref.en": "de-en",
        "newstest2019-fien-ref.en": "fi-en",
        "newstest2019-guen-ref.en": "gu-en",
        "newstest2019-kken-ref.en": "kk-en",
        "newstest2019-lten-ref.en": "lt-en",
        "newstest2019-ruen-ref.en": "ru-en",
        "newstest2019-zhen-ref.en": "zh-en",
    })

    if name == 'WMT17':
        dataset = WMT2017
    if name == 'WMT18':
        dataset = WMT2018
    if name == 'WMT19':
        dataset = WMT2019
    return dataset


def load_data(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines


def load_metadata(lp):
    files_path = []
    for root, directories, files in os.walk(lp):
        for file in files:
            if '.hybrid' not in file:
                raw = file.split('.')
                testset = raw[0]
                lp = raw[-1]
                system = '.'.join(raw[1:-1])
                files_path.append((os.path.join(root, file), testset, lp, system))
    return files_path


def df_append(metric, num_samples, lp, testset, system, score):
    return pd.DataFrame({'metric': [metric] * num_samples,
                         'lp': [lp] * num_samples,
                         'testset': [testset] * num_samples,
                         'system': [system] * num_samples,
                         'sid': [_ for _ in range(1, num_samples + 1)],
                         'score': score,
                         })

def print_sys_level_correlation(metric, data, lp_set, csv_file, f="DA-syslevel.csv"):
    results = pd.concat(data, ignore_index=True)
    results.to_csv(csv_file, sep='\t', index=False)
    del results['sid']
    results = results.groupby(['metric', 'lp', 'testset', 'system']).mean()
    results = results.reset_index()
    results.to_csv('scores.sys.score', sep='\t', index=False, header=False)

    humanScoresDict = dict()
    for line in open(f):
        lp, human, system = line.strip().split(" ")
        if lp not in humanScoresDict:
            humanScoresDict[lp] = dict()
        humanScoresDict[lp][system] = human

    lp_vs_pearson = dict()
    for lp in lp_set:
        humanScores = []
        ourScores = []
        for _, row in results.iterrows():
            if row["metric"] != metric or row["lp"] != lp:
                continue
            system = row["system"]
            score = row["score"]
            if lp not in humanScoresDict or system not in humanScoresDict[lp]:
                print("ERROR HUMAN SCORE not found", lp, system)
                continue
            ourScores.append(float(score))
            humanScores.append(float(humanScoresDict[lp][system]))
        if (len(humanScores) >= 2):
            pr = pearsonr(humanScores, ourScores)
            lp_vs_pearson[lp] = pr[0]
    print(metric, lp_vs_pearson)
