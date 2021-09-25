from __future__ import absolute_import, division, unicode_literals

import string

from nltk.parse.stanford import StanfordDependencyParser
import os
from apted import APTED, Config
import unicodedata
import csv
from mover.moverscore_v2 import word_mover_score
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import bert_score
from sacremoses import MosesDetokenizer
import torch
import time

from xmover.scorer import XMOVERScorer
import truecase
from laserembeddings import Laser

from transformers import *
import sentence_transformers
from sbert_wk.utils import generate_embedding
import jieba
import stanza
import argparse

import sys

from bleurt import score

from gensim.models.keyedvectors import KeyedVectors

import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='./data/raw/paws.tsv')
parser.add_argument('--output', type=str, default='out_paws.tsv')
args = parser.parse_args()

all_chars = (chr(i) for i in range(0x110000))
control_chars = ''.join(c for c in all_chars if unicodedata.category(c)[0] == 'C')
control_chars = '/\\0'

# Workaround for bug in bleurt (https://github.com/huggingface/datasets/issues/1727)
sys.argv = sys.argv[:1]

class DTED:


    """
    takes a pair of sentences (as an iterable) and produces the cosine similarity between the SBERT-WK embeddings for 
    those sentences
    """
    def sbert_wk_similarity(self, sentences, tokenizer, model, device):

        sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        features_input_ids = []
        features_mask = []
        for sent_ids in sentences_index:
            # Truncate if too long
            if len(sent_ids) > 128:
                sent_ids = sent_ids[:128]
            sent_mask = [1] * len(sent_ids)
            # Padding
            padding_length = 128 - len(sent_ids)
            sent_ids += ([0] * padding_length)
            sent_mask += ([0] * padding_length)
            # Length Check
            assert len(sent_ids) == 128
            assert len(sent_mask) == 128

            features_input_ids.append(sent_ids)
            features_mask.append(sent_mask)

        batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
        batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        model.zero_grad()

        with torch.no_grad():
            features = model(**inputs)[1]

        features = [layer_emb.cpu().numpy() for layer_emb in features]
        all_layer_embedding = []
        for i in range(features[0].shape[0]):
            all_layer_embedding.append(np.array([layer_emb[i] for layer_emb in features]))
        params = {'context_window_size': 2, 'layer_start': 4}
        embed_method = generate_embedding('ave_last_hidden', features_mask)
        embedding = embed_method.embed(params, all_layer_embedding)
        return embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[1])



    def metric_combination(self, a, b, alpha):
        return alpha[0] * np.array(a) + alpha[1] * np.array(b)



    """
       for a given tsv file with sentence pairs in english
       and semantic similarity scores for them, calculates and returns:
       BERTScore, MoverScore, SBERT, LASER, Syntactic score based on TED, morphological similarity scores, BLEU scores
       input_file: Path to input tsv file.
       embedding_file: Path to file with embeddings for morphological comparison.
    """

    def score_en(self, input_file):
        references = []
        hypothesis = []
        sbert_scores = []
        mover_scores = []
        sbert_wk_scores = []
        laser_scores = []
        labse_scores = []
        mSBERT_scores = []
        bleurt_scores = []
        mUSE_scores = []
        ref_det = []
        labels = []
        hyp_det = []
        laser = Laser()
        bleurt_scorer = score.BleurtScorer("bleurt-base-128")
        detok_en = MosesDetokenizer(lang='en')
        model_sbert = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')
        model_mSBERT = sentence_transformers.SentenceTransformer('stsb-xlm-r-multilingual')
        model_labse = sentence_transformers.SentenceTransformer('LaBSE')
        model_mUSE = sentence_transformers.SentenceTransformer('distiluse-base-multilingual-cased')
        torch.cuda.set_device(-1)
        device = torch.device("cuda", 0)
        config_wk = AutoConfig.from_pretrained('bert-base-uncased', cache_dir='./cache')
        config_wk.output_hidden_states = True
        tokenizer_wk = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
        model_wk = AutoModelWithLMHead.from_pretrained('bert-base-uncased', config=config_wk, cache_dir='./cache')
        model_wk.to(device)

        stanza.download('en', processors='tokenize,  pos', package='gum')
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        with open(input_file, encoding='utf-8', errors="surrogateescape") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                labels.append(row[0])
                references.append(row[1])
                hypothesis.append(row[2])
                print(row)
                ref_det.append(detok_en.detokenize(row[1].split(' ')))
                hyp_det.append(truecase.get_true_case(detok_en.detokenize(row[2].split(' '))))


                # LASER
                embedding = laser.embed_sentences([row[1], row[2]], lang='en')
                laser_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                laser_scores.append(laser_temp)

                # BLEURT
                scores = bleurt_scorer.score([row[1]], [row[2]])
                bleurt_scores.append(scores[0])

                # mUSE
                embedding = model_mUSE.encode([row[1], row[2]])
                sbert_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                mUSE_scores.append(sbert_temp)

                # LABSE
                embedding = model_labse.encode([row[1], row[2]])
                labse_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                labse_scores.append(labse_temp)

                # stsb xlm
                embedding = model_mSBERT.encode([row[1], row[2]])
                xlm_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                mSBERT_scores.append(xlm_temp)

                # SBERT
                embedding = model_sbert.encode([row[1], row[2]])
                sbert_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                sbert_scores.append(sbert_temp)

                sbert_wk_scores.append(self.sbert_wk_similarity([row[1], row[2]], tokenizer_wk, model_wk, device))

                mover_scores.append(word_mover_score([row[1]], [row[2]], idf_dict_ref, idf_dict_hyp,
                                                     stop_words=[], n_gram=1, remove_subwords=True)[0])

            del model_wk
            del model_sbert
            del model_mUSE
            _, _, bert_scores = bert_score.score(references, hypothesis, lang='eng',
                                                 rescale_with_baseline=True)
            bert_scores = bert_scores.tolist()
        return labels, references, hypothesis, \
               bert_scores, laser_scores, mUSE_scores, mover_scores, sbert_scores, sbert_wk_scores, labse_scores, mSBERT_scores, bleurt_scores

if __name__ == '__main__':
    dted = DTED()

    tmp = dted.score_en(args.input)

    with open(args.output, 'wt', encoding='utf-8', newline='') as output:
        tsv_writer = csv.writer(output, delimiter='\t')
        for i in range(len(tmp[0])):
            tsv_writer.writerow([a[i] for a in tmp if len(a) > 0])
