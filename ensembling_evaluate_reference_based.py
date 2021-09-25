import argparse
import os

import pandas as pd
from scipy.stats import pearsonr

import numpy as np
import tqdm

# System level
def get_corr_sys_da(metrics, df, corpus):
    lp_set = df.lp.unique()
    df = df.loc[df['metric'].isin(metrics)]

    del df['metric']
    # Ensembling
    df = df.groupby(['sid', 'lp', 'system']).mean()
    df = df.reset_index()
    # Get system scores
    del df['sid']
    df = df.groupby(['system', 'lp']).mean()
    df = df.reset_index()

    f = os.path.join(corpus,"DA-syslevel.csv")
    humanScoresDict = dict()
    humanScores = []
    with open(f) as hf:
        next(hf)
        for line in hf:
            lp, human, system = line.strip().split(" ")

            if lp not in humanScoresDict:
                humanScoresDict[lp] = dict()
            humanScoresDict[lp][system] = float(human)
            humanScores.append(float(human))

    humanMean = np.array(humanScores).mean()
    humanStd = np.array(humanScores).std()

    for lp in humanScoresDict.keys():
        for system in humanScoresDict[lp].keys():
           humanScoresDict[lp][system] = (humanScoresDict[lp][system]-humanMean)/humanStd

    lp_vs_pearson = dict()
    for lp in lp_set:
        humanScores = []
        ourScores = []
        for _, row in df.iterrows():
            if row["lp"] != lp:
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
    print(metrics, lp_vs_pearson)

# Segment level. Direct assessment
def get_corr_seg_da(metrics, df, corpus):
    lp_set = df.lp.unique()
    df = df.loc[df['metric'].isin(metrics)]

    del df['metric']
    # Ensembling
    df = df.groupby(['sid', 'lp', 'system']).mean()
    df = df.reset_index()

    f = os.path.join(corpus,"DA-seglevel.csv")
    humanScoresDict = dict()
    humanScores = []
    with open(f) as hf:
        next(hf)
        for line in hf:
            lp, _, system, sid, human = line.strip().split(" ")

            if lp not in humanScoresDict:
                humanScoresDict[lp] = dict()
            if system not in humanScoresDict[lp]:
                humanScoresDict[lp][system] = dict()
            sid = int(sid)
            humanScoresDict[lp][system][sid] = float(human)
            humanScores.append(float(human))

    humanMean = np.array(humanScores).mean()
    humanStd = np.array(humanScores).std()

    for lp in humanScoresDict.keys():
        for system in humanScoresDict[lp].keys():
            for sid in humanScoresDict[lp][system].keys():
                humanScoresDict[lp][system][sid] = (humanScoresDict[lp][system][sid]-humanMean)/humanStd

    lp_vs_pearson = dict()
    for lp in lp_set:
        humanScores = []
        ourScores = []
        ndf = df.loc[df.lp == lp]

        for _, row in ndf.iterrows():
            system = row["system"]
            sid = int(row["sid"])
            score = row["score"]

            if lp not in humanScoresDict or system not in humanScoresDict[lp] or sid not in humanScoresDict[lp][system]:
                # print("ERROR HUMAN SCORE not found", lp, system, sid)
                continue
            ourScores.append(float(score))
            humanScores.append(float(humanScoresDict[lp][system][sid]))
        if (len(humanScores) >= 2):
            pr = pearsonr(humanScores, ourScores)
            lp_vs_pearson[lp] = pr[0]
    print(metrics, lp_vs_pearson)

# Segment level relative ranking
def get_corr_seg_rr(metrics, df, betterWorseDict):
    lp_set = df.lp.unique()
    df = df.loc[df['metric'].isin(metrics)]

    del df['metric']
    # Ensembling
    df = df.groupby(['sid', 'lp', 'system']).mean()
    df = df.reset_index()

    lp_vs_kendall = dict()
    for lp in lp_set:
        if lp not in betterWorseDict:
            continue
        ndf = df.loc[df.lp == lp]
        system_sid = dict()

        for _, row in ndf.iterrows():
            system = row["system"]
            sid = int(row["sid"])
            score = row["score"]
            if system not in system_sid:
                system_sid[system] = dict()
            system_sid[system][sid] = score
        conc = 0
        disc = 0
        for sid in betterWorseDict[lp].keys():
            for pair in betterWorseDict[lp][sid]:
                better, worse = pair
                if better in system_sid and worse in system_sid and sid in system_sid[better] and sid in system_sid[worse]:
                    betterScore = system_sid[better][sid]
                    worseScore = system_sid[worse][sid]
                    if betterScore > worseScore:
                        conc += 1
                    else:
                        disc += 1
        conc = float(conc)
        disc = float(disc)
        if conc+disc != 0:
            lp_vs_kendall[lp] = (conc-disc)/(conc+disc)

    print(metrics, lp_vs_kendall)

def normalize_scores(scores):
    mean = np.array(scores).mean()
    std = np.array(scores).std()
    # print(mean,std)
    scores = [ (s-mean)/std for s in scores]
    return scores

# For relative ranking
def load_better_worse_dict(corpus):
    f = os.path.join(corpus,"RR-seglevel.csv")
    betterWorseDict = dict()
    with open(f) as hf:
        next(hf)
        for line in hf:
            lp,_, sid, better, worse = line.strip().split(" ")
            if better.endswith("zh-en"):
                better = better[:better.rindex(".")]
            if worse.endswith("zh-en"):
                worse = worse[:worse.rindex(".")]
            sid = int(sid)

            if lp not in betterWorseDict:
                betterWorseDict[lp] = dict()
            if sid not in betterWorseDict[lp]:
                betterWorseDict[lp][sid] = []
            betterWorseDict[lp][sid].append((better, worse))
    return betterWorseDict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=None, type=str, help="comma separated list of metrics")
    parser.add_argument("--eval", default="sys", type=str, help="sys or seg")
    parser.add_argument("--scores", type=str, default="results/sentence_pair.scores")
    parser.add_argument("--corpus", type=str, default="WMT19")
    parser.add_argument("--rr", type=bool, default=False)
    parser.add_argument("--avg", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    df = pd.read_csv(args.scores, sep="\t")
    del df['testset']

    # Normalize scores
    for metric in df["metric"].unique():
        ndf = df.loc[df['metric'] == metric]
        scores = [row["score"] for _, row in ndf.iterrows()]
        scores = normalize_scores(scores)
        for i,(id, _ ) in enumerate(ndf.iterrows()):
            df.at[id, "score"] = scores[i]

    # Parse metrics
    if not args.avg and args.metrics is None:
        metrics = [ [m] for m in  df["metric"].unique()]
    elif not args.avg:
        metric = [ [m] for m in  args.metrics.split(",")]
    elif args.avg and args.metrics is None:
        metric = [df["metric"].unique()]
    else:
        metrics = [args.metrics.split(",")]

    print(metrics)

    # Evaluate
    if args.eval == "seg" and args.rr:
        betterWorseDict = load_better_worse_dict(args.corpus)
        for metric in tqdm.tqdm(metrics):
            get_corr_seg_rr(metric, df, betterWorseDict)
    elif args.eval == "seg":
        for metric in tqdm.tqdm(metrics):
            get_corr_seg_da(metric, df, args.corpus)
    elif args.eval == "sys":
        for metric in tqdm.tqdm(metrics):
            get_corr_sys_da(metric, df, args.corpus)
