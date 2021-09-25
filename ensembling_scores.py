import argparse
import os
import traceback
import datetime

import pandas as pd
import tqdm
from nltk.translate.bleu_score import SmoothingFunction

from kobe.eval_main import read_all_annotations, get_scores_dictionary, KOBE, \
    get_score_for_annotated_sentence, SOURCE, ANNOTATED_SENTENCE, ENTITIES
from laserembeddings import Laser
from xmover2.mt_utils import find_corpus, load_data, load_metadata, df_append, print_sys_level_correlation
from xmover2.scorer import XMOVERScorer

import numpy as np

from sentence_transformers import SentenceTransformer, util
import nltk
import spacy_udpipe


"""

:param msbert: Sentence transformer model
:param source: Input sentences (list of str)
:param translations: Translated sentences
:returns: List of scores
"""
def get_msbert_scores(msbert, source, translations, batch_size=32):
    scores = []
    for i in range(0,len(source),batch_size):
        src_embeddings = msbert.encode(source[i:min(i+batch_size,len(source))], convert_to_tensor=True, batch_size=batch_size, device="cuda:0")
        trs_embeddings = msbert.encode(translations[i:min(i+batch_size,len(source))], convert_to_tensor=True, batch_size=batch_size, device="cuda:0")
        scores += [ util.pytorch_cos_sim(src_embeddings[j], trs_embeddings[j]).item() for j in range(len(src_embeddings))]
    return scores

"""
Weighted sum for xms
"""
def metric_combination(a, b, alpha):
    return alpha[0]*np.array(a) + alpha[1]*np.array(b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="WMT19", type=str)
    parser.add_argument("--csv", default='sentence_pair.score', type=str)
    parser.add_argument("--metrics", default=None, type=str)
    parser.add_argument("--lang", default=None, type=str)
    args = parser.parse_args()
    corpus = args.corpus
    metrics = None if args.metrics is None else args.metrics.lower().split(",")

    print("Corpus", corpus)

    # Load models
    if metrics is None or "xms" in metrics:
        scorer = XMOVERScorer("bert-base-multilingual-cased", "gpt2", False)

    if metrics is None or "laser" in metrics:
        laser_encoder = Laser()

    if metrics is None or "msbert" in metrics:
        msbert1 = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        msbert2 = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')
        msbert3 = SentenceTransformer('LaBSE')

    if metrics is None or "distilbert-quora" in metrics:
        msbert4 = SentenceTransformer('quora-distilbert-multilingual')

    # Dataset path
    dataset = find_corpus(corpus)

    # List which stores the calculated scores (list of df)
    wmt_xmoverscores = []

    # Iterate over each language pair in the dataset
    for pair in tqdm.tqdm(dataset.items()):
        # lp = Language-pair as string e.g 'en-de'
        reference_path, lp = pair
        try:

            src, tgt = lp.split('-')
            if args.lang is not None and src not in args.lang:
                continue
            # List of reference sentences. Only for bleu
            references = load_data(os.path.join(corpus,'references/', reference_path))

            # List of source sentences
            source_path = reference_path.replace('ref', 'src')
            source_path = source_path.split('.')[0] + '.' + src
            source = load_data(os.path.join(corpus,'source', source_path))

            # List of (Path, testset, lp, system) for this language pair
            # Each system has its own entry
            # system = system name
            # Path = Path to the system outputs
            all_meta_data = load_metadata(os.path.join(corpus,'system-outputs', lp))

            # Update models which need information about the current language pair
            if metrics is None or "xms" in metrics:
                scorer.setLanguagePair(src, tgt)

            # KoBe needs own dataset because of srl annotations
            if (metrics is None or "kobe" in metrics) and corpus == "WMT19":
                kobe_lp_annotations = read_all_annotations("kobe-data", [lp])[lp]
            else:
                kobe_lp_annotations = None

            # Iterate over each system for this language pair
            for i in range(len(all_meta_data)):
                path, testset, lp_, system = all_meta_data[i]
                if lp != lp_:
                    continue

                # All translations for a single system for a certain language pair
                # List of translations
                translations = load_data(path)
                num_samples = len(references)
                df_system = pd.DataFrame(columns=('metric', 'lp', 'testset', 'system', 'sid', 'score'))

                # Sentence Transformers
                if metrics is None or "msbert" in metrics:
                    # Calculate scores
                    msbert_scores = get_msbert_scores(msbert1, source, translations)
                    # Add scores to dataframe
                    wmt_xmoverscores.append(df_append('msbert-distil', num_samples, lp, testset, system, msbert_scores))

                    msbert_scores2 = get_msbert_scores(msbert2, source, translations)
                    wmt_xmoverscores.append(df_append('msbert-xlm', num_samples, lp, testset, system, msbert_scores2))
                    msbert_scores1 = get_msbert_scores(msbert3, source, translations)
                    wmt_xmoverscores.append(df_append('msbert-labse', num_samples, lp, testset, system, msbert_scores1))

                if metrics is None or "distilbert-quora" in metrics:
                    quora_scores = get_msbert_scores(msbert4, source, translations)
                    wmt_xmoverscores.append(df_append('distilbert-quora', num_samples, lp, testset, system, quora_scores))

                # XMOVERSCORE
                if metrics is None or "xms" in metrics:
                    try:
                        d = scorer.compute_xmoverscore(source, translations, bs=64)
                        # d [projection: CLP or UMD] [n-gram: 1 or 2]
                        xmoverscores1 = d["CLP"][1]
                        wmt_xmoverscores.append(df_append('xmoverscore1-CLP', num_samples, lp, testset, system, xmoverscores1))
                        wmt_xmoverscores.append(df_append('xmoverscore2-CLP', num_samples, lp, testset, system, d["CLP"][2]))
                        wmt_xmoverscores.append(df_append('xmoverscore1-UMD', num_samples, lp, testset, system, d["UMD"][1]))
                        wmt_xmoverscores.append(df_append('xmoverscore2-UMD', num_samples, lp, testset, system, d["UMD"][2]))


                        lm_perplexity = scorer.compute_perplexity(translations, bs=1)
                        xmoverscores1_lm = metric_combination(xmoverscores1, lm_perplexity, [1, 0.1])
                        xmoverscores2_lm = metric_combination(d["CLP"][2], lm_perplexity, [1, 0.1])
                        xmoverscores3_lm = metric_combination(d["UMD"][1], lm_perplexity, [1, 0.1])
                        xmoverscores4_lm = metric_combination(d["UMD"][2], lm_perplexity, [1, 0.1])

                        wmt_xmoverscores.append(df_append('xmoverscore1-CLP+LM', num_samples, lp, testset, system, xmoverscores1_lm))
                        wmt_xmoverscores.append(df_append('xmoverscore2-CLP+LM', num_samples, lp, testset, system, xmoverscores2_lm))
                        wmt_xmoverscores.append(df_append('xmoverscore1-UMD+LM', num_samples, lp, testset, system, xmoverscores3_lm))
                        wmt_xmoverscores.append(df_append('xmoverscore2-UMD+LM', num_samples, lp, testset, system, xmoverscores4_lm))
                    except:
                        print("Error XMS", lp, system)
                        print(traceback.format_exc())
                        print()
                        print()

               
                # KoBe
                # Calculate recall for each sentence pair instead of the whole system like in the paper
                if metrics is None or "kobe" in metrics:
                    try:
                        if kobe_lp_annotations is not None:
                            kobe_scores = [get_score_for_annotated_sentence( kobe_lp_annotations[SOURCE][ANNOTATED_SENTENCE][j],   kobe_lp_annotations[system][ANNOTATED_SENTENCE][j]) for j in range(num_samples)]
                            wmt_xmoverscores.append(df_append("kobe", num_samples, lp, testset, system, kobe_scores))
                    except:
                        print("Error KoBe", lp, system)
                        print(traceback.format_exc())
                        print()
                        print()

                # LASER
                if metrics is None or "laser" in metrics:
                    try:
                        embedding1 = laser.embed_sentences(source, lang=src)
                        embedding2 = laser.embed_sentences(translations, lang=tgt)
                        for e1,e2 in zip(embedding,embedding2):
                            laser_temp = e1.dot(e2) / np.linalg.norm(e1) / np.linalg.norm(
                                e2)
                            laser_scores.append(laser_temp)
                        wmt_xmoverscores.append(df_append("laser", num_samples, lp, testset, system, laser_scores))
                    except:
                        print("Error laser", lp, system)
                        print(traceback.format_exc())
                        print()
                        print()

                # BLEU
                if metrics is None or "bleu" in metrics:
                    try:
                        spacy_udpipe.download(tgt)
                        tgt_udpipe = spacy_udpipe.load(tgt)
                        tokenized_refs = [[t.text for t in tgt_udpipe(ref.lower()) ]for ref in references]
                        tokenized_trans =  [[t.text for t in tgt_udpipe(trans.lower()) ]for trans in translations]
                        bleu_scores = [nltk.translate.bleu_score.sentence_bleu([tokenized_refs[j]], tokenized_trans[j], smoothing_function=SmoothingFunction().method1) for j in range(len(references))]
                        wmt_xmoverscores.append(df_append("bleu", num_samples, lp, testset, system, bleu_scores))
                    except:
                        print("Error BLEU", lp, system)
                        print(traceback.format_exc())
                        print()
                        print()

                # Save results after each language pair. Just in case something goes wrong...
                results = pd.concat(wmt_xmoverscores, ignore_index=True)
                timeNow = str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
                checkpointPath = "checkpoint/"+timeNow+".score"
                print("Saved checkpoint "+checkpointPath)
                results.to_csv(checkpointPath, sep='\t', index=False)

        except Exception as e:
            print("Error", lp)
            print(traceback.format_exc())
            print()
            print()

    # Print system level pearson correlation
    # All scores are saved at args.csv
    for metric in pd.concat(wmt_xmoverscores, ignore_index=True).metric.unique():
        print_sys_level_correlation(metric, wmt_xmoverscores, list(dataset.values()),args.csv,  os.path.join(corpus,'DA-syslevel.csv'))
