import argparse
import os
import traceback
import datetime
from collections import defaultdict

import pandas as pd
import tqdm
from nltk.translate.bleu_score import SmoothingFunction

from laserembeddings import Laser

from kobe.eval_main import read_all_annotations, get_scores_dictionary, KOBE, \
    get_score_for_annotated_sentence, SOURCE, ANNOTATED_SENTENCE, ENTITIES
from mover.moverscore_v2 import word_mover_score
from xmover2.mt_utils import find_corpus, load_data, load_metadata, df_append, print_sys_level_correlation
from xmover2.scorer import XMOVERScorer

import numpy as np

from sentence_transformers import SentenceTransformer, util
import nltk
import spacy_udpipe

import bert_score


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

    if metrics is None or "msbert" in metrics:
        msbert1 = SentenceTransformer('stsb-xlm-r-multilingual')
    if metrics is None or "sbert" in metrics:
        model_sbert = SentenceTransformer('bert-base-nli-mean-tokens')
    if metrics is None or "mUSE" in metrics:
        model_mUSE = SentenceTransformer('distiluse-base-multilingual-cased')
    if metrics is None or "laser" in metrics:
        laser = Laser()


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
            # List of reference sentences.
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
                    msbert_scores = get_msbert_scores(msbert1, references, translations)
                    # Add scores to dataframe
                    wmt_xmoverscores.append(df_append('msbert', num_samples, lp, testset, system, msbert_scores))


                # Sentence Transformers
                if metrics is None or "bertscore" in metrics:
                    _, _, bert_scores = bert_score.score(references, translations, lang='eng',
                                                 rescale_with_baseline=True)
                    bert_scores = bert_scores.tolist()
                    # Add scores to dataframe
                    wmt_xmoverscores.append(df_append('bertscore', num_samples, lp, testset, system, bert_scores))

                if metrics is None or "moverscore" in metrics:
                    idf_dict_hyp = defaultdict(lambda: 1.)
                    idf_dict_ref = defaultdict(lambda: 1.)
                    mover_scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp,
                                                     stop_words=[], n_gram=1, remove_subwords=True)
                    wmt_xmoverscores.append(df_append('moverscore', num_samples, lp, testset, system, mover_scores))

                if metrics is None or "sbert" in metrics:
                    sbert_scores = []
                    for ref, trans in zip(references, translations):
                        embedding = model_sbert.encode([ref,trans])
                        sbert_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                            embedding[1])
                        sbert_scores.append(sbert_temp)
                    wmt_xmoverscores.append(df_append('sbert', num_samples, lp, testset, system, sbert_scores))

                if metrics is None or "laser" in metrics:
                    laser_scores = []
                    for ref, trans in zip(references, translations):
                        embedding = laser.embed_sentences([ref, trans], lang='en')
                        laser_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                            embedding[1])
                        laser_scores.append(laser_temp)
                    wmt_xmoverscores.append(df_append('laser', num_samples, lp, testset, system, laser_scores))

                if metrics is None or "mUSE" in metrics:
                    mUSE_scores = []
                    for ref, trans in zip(references, translations):
                        embedding = model_mUSE.encode([ref, trans])
                        mUSE_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                            embedding[1])
                        mUSE_scores.append(mUSE_temp)
                    wmt_xmoverscores.append(df_append('mUSE', num_samples, lp, testset, system, mUSE_scores))





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
