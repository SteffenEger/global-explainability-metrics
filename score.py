from __future__ import absolute_import, division, unicode_literals

import string

from nltk.parse.stanford import StanfordDependencyParser
import os
from apted import APTED, Config
import unicodedata
import csv
from mover.moverscore_v2 import word_mover_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
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

parser.add_argument('--input', type=str, default='./data/raw/wmt_en_full.tsv')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--embeddings', type=str, default='./data/morph/embeddings/retrofitted/retro_wmt_en.tsv')
parser.add_argument('--output', type=str, default='output.tsv')
parser.add_argument('--translations', type=str, default=None)
args = parser.parse_args()

all_chars = (chr(i) for i in range(0x110000))
control_chars = ''.join(c for c in all_chars if unicodedata.category(c)[0] == 'C')
control_chars = '/\\0'

# Workaround for bug in bleurt (https://github.com/huggingface/datasets/issues/1727)
sys.argv = sys.argv[:1]

"""
Custom configuration for comparing nodes in nltk parse tree
"""


class CustomConfig(Config):
    def rename(self, node1, node2):
        """ignore renaming nodes"""
        return 0

    def children(self, node):
        """Get all children"""
        return node if not isinstance(node, str) else []


class DTED:
    def __init__(self):
        os.environ['STANFORD_PARSER'] = 'stanford-parser-full-2015-04-20'
        os.environ['STANFORD_MODELS'] = 'stanford-parser-full-2015-04-20'

        self.dep_parser_en = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
                                                      encoding='utf-8')
        self.dep_parser_de = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz",
                                                      encoding='utf-8')
        self.dep_parser_zh = StanfordDependencyParser(
            model_path="edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz",
            encoding='utf-8')

    def get_avg_fasttext_embedding_for_sentence(self, words, fasttext_model):
        avg_sent = None
        for word in words:
            word = word.strip().lower()
            if fasttext_model.has_index_for(word):
                if avg_sent is None:
                    avg_sent = fasttext_model[word]
                else:
                    avg_sent = np.vstack((avg_sent, fasttext_model[word]))
        if avg_sent is None:
            return None
        return avg_sent.mean(axis=0)

    def score_fasttext(self, sent_1, sent_2, fasttext_model_1, fasttext_model_2, lang):
        words_1 = word_tokenize(sent_1, language="english")
        if lang == "en":
            words_2 = word_tokenize(sent_2, language="english")
        elif lang == "de":
            words_2 = word_tokenize(sent_2, language="german")
        else:
            words_2 = list(jieba.cut(sent_2))

        avg_sent_1 = self.get_avg_fasttext_embedding_for_sentence(words_1, fasttext_model_1)
        avg_sent_2 = self.get_avg_fasttext_embedding_for_sentence(words_2, fasttext_model_2)

        if avg_sent_1 is None or avg_sent_2 is None or avg_sent_1.size <= 1 or avg_sent_2.size <= 1:
            return 0

        sim = avg_sent_1.dot(avg_sent_2) / np.linalg.norm(avg_sent_1) / np.linalg.norm(
                    avg_sent_2)

        return sim

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

    """
    calculates a cross-lingual BLEU score between two sentences by translating the non-English sentence into English
    and then calculating normal BLEU. This can be done word-for-word or as complete sentences, as given by
    hypothesis: English sentence
    reference: non-English sentence
    lang: language of the non-English sentence
    strategy: "word" or "sentence", depending on desired translation strategy
    translator: translator object
    """
    def score_bleu_cross(self, hypothesis, ref_translation):
        bleu_1 = sentence_bleu([ [ r.lower() for r in ref_translation]], word_tokenize(hypothesis.lower()), (1., 0.))
        return bleu_1

    """
    calculates a Morphological similarity score between two sentences using word embeddings that have previously been 
    enriched with morphological information (given as embedding_file). These are then averaged over each sentence and 
    compared.
    """
    def score_morph(self, sent_1, sent_2, embedding_file, pipe_1, pipe_2):
        model = {}
        with open(embedding_file, encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter=' ')
            for row in reader:
                model[row[0]] = np.array([np.float(j) for j in row[1:-1]])
        sent_1 = [word for sen in pipe_1(sent_1).sentences for word in sen.words]
        sent_2 = [word for sen in pipe_2(sent_2).sentences for word in sen.words]
        sent_1_emb = []
        sent_2_emb = []
        for word in sent_1:
            if word.text.lower() not in model.keys():
                continue
            sent_1_emb.append(model[word.text.lower()])
        for word in sent_2:
            if word.text.lower() not in model.keys():
                continue
            sent_2_emb.append(model[word.text.lower()])
        sent_1_emb = np.mean(sent_1_emb, axis=0)
        sent_2_emb = np.mean(sent_2_emb, axis=0)
        if np.linalg.norm(sent_1_emb) > 0 and np.linalg.norm(sent_2_emb) > 0:
            score = sent_1_emb.dot(sent_2_emb) / np.linalg.norm(sent_1_emb) / np.linalg.norm(sent_2_emb)
        else:
            score = 0
        return score

    def metric_combination(self, a, b, alpha):
        return alpha[0] * np.array(a) + alpha[1] * np.array(b)

    """
    for a given tsv file with sentence pairs between english and a given language (de or zh)
    and semantic similarity scores for them, calculates and returns:
    SBERT, LASER, Syntactic score based on TED
    input_file: Path to input tsv file.
    lang: language of the non-English sentences given. (zh or de)
    embedding_file: Path to file with embeddings for morphological comparison.
    """

    def score_cross(self, input_file, lang, embedding_file, translations_file):
        references = []
        hypothesis = []
        sem_scores = []
        syn_scores = []
        mUSE_scores = []
        morph_scores = []
        fasttext_scores = []
        bleu_scores_word = []
        bleu_scores_sentence = []
        labse_scores = []
        mSBERT_scores = []
        laser_scores = []
        ref_det = []
        hyp_det = []
        laser = Laser()
        # Read translations
        with open(translations_file, "rb") as in_file:
            translations_dict = pickle.load(in_file)
            translations_word = translations_dict["word"]
            translations_sentence = translations_dict["sentence"]

        detok_en = MosesDetokenizer(lang='en')
        detok_cross = MosesDetokenizer(lang=lang)
        model_mUSE = sentence_transformers.SentenceTransformer('distiluse-base-multilingual-cased')
        model_mSBERT = sentence_transformers.SentenceTransformer('stsb-xlm-r-multilingual')
        model_labse = sentence_transformers.SentenceTransformer('LaBSE')
        stanza.download('en', processors='tokenize,  pos', package='gum')
        en_pipe = stanza.Pipeline('en', processors='tokenize,  pos', package='gum')
        if lang == "de":
            stanza.download('de', processors='tokenize,  pos', package='hdt')
            de_pipe = stanza.Pipeline('de', processors='tokenize,  pos', package='hdt')
        else:
            stanza.download('zh', processors='tokenize, pos', package='gsdsimp')
            zh_pipe = stanza.Pipeline('zh', processors='tokenize, pos', package='gsdsimp', tokenize_with_jieba=True)
        with open(input_file, encoding='utf-8', errors="surrogateescape") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            i = 0
            for row in reader:
                hypothesis.append(row[1])
                references.append(row[2])
                print(row)
                ref_det.append(detok_cross.detokenize(row[2].split(' ')))
                hyp_det.append(truecase.get_true_case(detok_en.detokenize(row[1].split(' '))))
                sem_scores.append(row[0])
                parser_1 = dted.dep_parser_en
                if lang == "zh":
                    ref_temp = ' '.join(jieba.cut(row[2]))
                    parser_2 = dted.dep_parser_zh
                else:

                    ref_temp = row[2]
                    parser_2 = dted.dep_parser_de
                syn_score_temp = dted.process_batch([row[1]], [ref_temp], parser_1, parser_2)
                syn_scores.append(syn_score_temp)

                # LASER

                embedding = laser.embed_sentences([ref_temp, row[1]], lang=[lang, 'en'])
                laser_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                laser_scores.append(laser_temp)

                # mUSE
                embedding = model_mUSE.encode([ref_temp, row[1]])
                sbert_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                mUSE_scores.append(sbert_temp)

                # LABSE
                embedding = model_labse.encode([ref_temp, row[1]])
                labse_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                labse_scores.append(labse_temp)

                # stsb xlm
                embedding = model_mSBERT.encode([ref_temp, row[1]])
                xlm_temp = embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(
                    embedding[1])
                mSBERT_scores.append(xlm_temp)

                bleu_temp_1 = self.score_bleu_cross(row[1], translations_word[row[2]])
                bleu_temp_2 = self.score_bleu_cross(row[1], translations_sentence[row[2]])
                bleu_scores_word.append(bleu_temp_1)
                bleu_scores_sentence.append(bleu_temp_2)
                if lang == "de":
                    morph_scores.append(self.score_morph(row[1], row[2], embedding_file, en_pipe, de_pipe))

                fasttext_scores.append(0)
                i += 1
            del model_mUSE, en_pipe
            torch.cuda.empty_cache()

            scorer = XMOVERScorer('bert-base-multilingual-cased', 'gpt2', False)
            device = 'cuda:0'
            temp = np.loadtxt('./xmover/mapping/europarl-v7.' + lang + '-' + 'en' + '.2k.12.BAM.map')
            projection = torch.tensor(temp, dtype=torch.float).to(device)
            temp = np.loadtxt('./xmover/mapping/europarl-v7.' + lang + '-' + 'en' + '.2k.12.GBDD.map')
            bias = torch.tensor(temp, dtype=torch.float).to(device)
            lm_scores = scorer.compute_perplexity(hypothesis, bs=1)
            xm_scores = scorer.compute_xmoverscore('CLP', projection, bias, ref_det, hyp_det, ngram=1, bs=64)
            xm_scores_lm = self.metric_combination(xm_scores, lm_scores, [1, 0.1])

        return references, hypothesis, sem_scores, syn_scores, morph_scores, bleu_scores_word, bleu_scores_sentence, fasttext_scores, \
               laser_scores, mUSE_scores, xm_scores_lm, labse_scores, mSBERT_scores
        #return references, hypothesis, sem_scores, syn_scores, morph_scores, bleu_scores_word, bleu_scores_sentence, \
        #       laser_scores, sbert_scores, xm_scores_lm, labse_scores, mSBERT_scores

    """
       for a given tsv file with sentence pairs in english
       and semantic similarity scores for them, calculates and returns:
       BERTScore, MoverScore, SBERT, LASER, Syntactic score based on TED, morphological similarity scores, BLEU scores
       input_file: Path to input tsv file.
       embedding_file: Path to file with embeddings for morphological comparison.
    """

    def score_en(self, input_file, embedding_file):
        references = []
        hypothesis = []
        sem_scores = []
        syn_scores = []
        fasttext_scores = []
        sbert_scores = []
        morph_scores = []
        bleu_1_scores = []
        mover_scores = []
        sbert_wk_scores = []
        laser_scores = []
        labse_scores = []
        mSBERT_scores = []
        bleurt_scores = []
        mUSE_scores = []
        ref_det = []
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
        en_pipe = stanza.Pipeline('en', processors='tokenize,  pos', package='gum')

        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        with open(input_file, encoding='utf-8', errors="surrogateescape") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            i = 0
            for row in reader:
                references.append(row[1])
                hypothesis.append(row[2])
                print(row)
                ref_det.append(detok_en.detokenize(row[1].split(' ')))
                hyp_det.append(truecase.get_true_case(detok_en.detokenize(row[2].split(' '))))
                sem_scores.append(row[0])

                syn_score_temp = dted.process_batch([row[1]], [row[2]], dted.dep_parser_en, dted.dep_parser_en)
                syn_scores.append(syn_score_temp)

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
                bleu_1_scores.append(sentence_bleu([word_tokenize(row[1].lower())], word_tokenize(row[2].lower()), (1, 0)))

                morph_scores.append(self.score_morph(row[1], row[2], embedding_file, en_pipe, en_pipe))
                i += 1
                mover_scores.append(word_mover_score([row[1]], [row[2]], idf_dict_ref, idf_dict_hyp,
                                                     stop_words=[], n_gram=1, remove_subwords=True)[0])

                fasttext_scores.append(0)


            del model_wk
            del model_sbert
            del model_mUSE
            _, _, bert_scores = bert_score.score(references, hypothesis, lang='eng',
                                                 rescale_with_baseline=True)
            bert_scores = bert_scores.tolist()
        return references, hypothesis, sem_scores, syn_scores, morph_scores, bleu_1_scores, fasttext_scores, \
               bert_scores, laser_scores, mUSE_scores, mover_scores, sbert_scores, sbert_wk_scores, labse_scores, mSBERT_scores, bleurt_scores
        #return references, hypothesis, sem_scores, syn_scores, morph_scores, bleu_1_scores,  \
        #       bert_scores, laser_scores, mover_scores, sbert_scores, sbert_wk_scores, labse_scores, mSBERT_scores, bleurt_scores

    def count_nodes(self, tree):
        if isinstance(tree, str):
            return 1
        return 1 + sum(self.count_nodes(child) for child in tree)

    """
    Compare the parse trees of two sentences
    reference: Tree sentence one
    hypothesis: Tree sentence two
    """

    def compare(self, reference_tree, hypothesis_tree):
        apted = APTED(reference_tree, hypothesis_tree, CustomConfig())
        edit_distance = apted.compute_edit_distance()
        nh = self.count_nodes(hypothesis_tree)
        nr = self.count_nodes(reference_tree)

        return 1 - edit_distance / (nh + nr)

    """
    Parse references and hypotheses into trees and compute the average edit distance
    references: list of string: reference sentences
    hypothesis: list of string: hypothesis sentences
    """

    def process_batch(self, references, hypothesis, ref_parser, hyp_parser):
        try:

            new_refs = []
            new_hyps = []
            for ref, hyp in zip(references, hypothesis):
                for p in string.punctuation:
                    ref = ref.replace(p, " ")
                    hyp = hyp.replace(p, " ")
                new_refs.append(ref)
                new_hyps.append(hyp)

            reference_trees = ref_parser.raw_parse_sents(new_refs)
            hypothesis_trees = hyp_parser.raw_parse_sents(new_hyps)

            score = 0

            for reference, hypothesis in zip(reference_trees, hypothesis_trees):
                try:
                    ref_temp = next(reference)
                    hyp_temp = next(hypothesis)
                    score += self.compare(ref_temp.tree(), hyp_temp.tree())

                except TypeError:
                    pass
        except Exception:
            return 0

        return score


if __name__ == '__main__':
    dted = DTED()

    if args.lang == 'en':
        tmp = dted.score_en(args.input, args.embeddings)
    else:
        tmp = dted.score_cross(args.input, args.lang, args.embeddings, args.translations)

    with open(args.output, 'wt', encoding='utf-8', newline='') as output:
        tsv_writer = csv.writer(output, delimiter='\t')
        for i in range(len(tmp[0])):
            tsv_writer.writerow([a[i] for a in tmp if len(a) > 0])
