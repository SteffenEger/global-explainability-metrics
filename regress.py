import csv
from collections import defaultdict

import numpy as np
import statsmodels.api as sm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, default='wmt_mono_final.tsv')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--bleu_strat', type=str, default='sentence')
args = parser.parse_args()

if __name__ == '__main__':
    params = defaultdict(lambda: [])

    with open(args.input_file, encoding='utf-8', errors="surrogateescape") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        if args.lang == 'en':
            score_keys = ["sem", "syn", "morph", "bleu", "fasttext", "bert", "laser", "mUSE", "mover", "sbert", "sbert_wk",
                          "labse", "mSBERT", "bleurt"]
        elif args.lang == "de":
            score_keys = ["sem", "syn", "morph", "bleu-w", "bleu-s", "fasttext", "laser", "mUSE", "xmover", "labse",
                          "mSBERT"]
        else:
            score_keys = ["sem", "syn", "bleu-w", "bleu-s", "fasttext", "laser", "mUSE", "xmover", "labse", "mSBERT"]
        # if args.lang == 'en':
        #     score_keys = ["sem", "syn", "morph", "bleu", "bert", "laser", "mover", "sbert", "sbert_wk",
        #                   "labse", "mSBERT", "bleurt"]
        # elif args.lang == "de":
        #     score_keys = ["sem", "syn", "morph", "bleu-w", "bleu-s", "laser", "sbert", "xmover", "labse",
        #                   "mSBERT"]
        # else:
        #     score_keys = ["sem", "syn", "bleu-w", "bleu-s",  "laser", "sbert", "xmover", "labse", "mSBERT"]
        for row in reader:
            for r, key in enumerate(score_keys):
                params[key].append(float(row[r + 2]))

        for key in params.keys():
            params[key] = np.reshape(np.array(params[key]).astype(np.float), (-1, 1))
            # Normalize parameters
            params[key] = np.divide(np.subtract(params[key], np.mean(params[key])), np.std(params[key]))
            # params_sq[key] = np.square(params[key])
            # params_sq[key] = np.divide(np.subtract(params_sq[key], np.mean(params_sq[key])), np.std(params_sq[key]))

        # for i in range(len(params["sem"])):
        #     s = ""
        #     for k in params.keys():
        #         s+= str(params[k][i]) + " "
        #
        #     print(s)

        # calculate VIF scores and linear regression for the parameters.
        # Only differ in which parameters are used

        # if args.lang == 'en':
        #     regressors = ['sem', 'syn', 'bleu', 'morph', "fasttext"]
        #     metrics = ['sbert', 'laser', "mUSE", 'mover', 'sbert_wk', 'bert', 'bleurt', 'labse', 'mSBERT']
        # if args.lang == 'de':
        #     regressors = ['sem', 'syn', 'bleu-w' if args.bleu_strat == "word" else "bleu-s", 'morph', 'fasttext']
        #     metrics = ['mUSE', 'laser', 'xmover', 'labse', 'mSBERT']
        # if args.lang == 'zh':
        #     regressors = ['sem', 'syn', 'bleu-w' if args.bleu_strat == "word" else "bleu-s", "fasttext"]
        #     metrics = ['mUSE', 'laser', 'xmover', 'labse', 'mSBERT']
        if args.lang == 'en':
            regressors = ['sem', 'syn', 'bleu', 'morph']
            metrics = ['sbert', 'laser', 'mUSE',  'mover', 'sbert_wk', 'bert', 'bleurt', 'labse', 'mSBERT']
        if args.lang == 'de':
            regressors = ['sem', 'syn', 'bleu-w' if args.bleu_strat == "word" else "bleu-s", 'morph']
            metrics = ['mUSE', 'laser', 'xmover', 'labse', 'mSBERT']
        if args.lang == 'zh':
            regressors = ['sem', 'syn', 'bleu-w' if args.bleu_strat == "word" else "bleu-s"]
            metrics = ['mUSE', 'laser', 'xmover', 'labse', 'mSBERT']
        ck = np.column_stack([params[key] for key in regressors])
        cc = np.corrcoef(ck, rowvar=False)
        VIF = np.linalg.inv(cc)
        print("VIF values for ", " ".join(regressors))
        print(VIF.diagonal())
        all_params = np.concatenate([ params[key] for key in regressors], axis=1)
        for key in metrics:
            reg_xmover_4d = sm.OLS(params[key], all_params).fit()
            pred = reg_xmover_4d.predict(all_params)
            print("Summary for {}:".format(key))
            print(reg_xmover_4d.summary())
