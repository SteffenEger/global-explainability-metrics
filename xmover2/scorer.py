import spacy_udpipe
import truecase
from mosestokenizer import MosesDetokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertModel, BertTokenizer, BertConfig
from xmover2.score_utils import word_mover_score, lm_perplexity
import numpy as np
import torch
import math

class XMOVERScorer:

    def __init__(
        self,
        model_name=None,
        lm_name=None,
        do_lower_case=False,      
        device='cuda:0'
    ):        
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True, return_dict=False)

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)        
        
        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)        
        self.lm.to(device)

    def compute_xmoverscore(self,  source, translations, bs):
        with MosesDetokenizer(self.src) as detokenize:
            source = [detokenize(s.split(' ')) for s in source]
        with MosesDetokenizer(self.tgt) as detokenize:
            translations = [detokenize(s.split(' ')) for s in translations]

        translations = [truecase.get_true_case(s) for s in translations]
        return word_mover_score( self.projection, self.bias, self.model, self.tokenizer, source, translations,  batch_size=bs)

    def compute_perplexity(self, translations, bs):
        with MosesDetokenizer(self.tgt) as detokenize:
            translations = [detokenize(s.split(' ')) for s in translations]

        translations = [truecase.get_true_case(s) for s in translations]
        return lm_perplexity(self.lm, translations, self.lm_tokenizer, batch_size=bs)


    def setLanguagePair(self, src,tgt):
        device='cuda:0'
        temp = np.loadtxt('mapping/europarl-v7.' + src + '-' + tgt + '.2k.12.BAM.map')
        self.projection = torch.tensor(temp, dtype=torch.float).to(device)

        temp = np.loadtxt('mapping/europarl-v7.' + src + '-' + tgt + '.2k.12.GBDD.map')
        self.bias = torch.tensor(temp, dtype=torch.float).to(device)

        self.src = src
        self.tgt = tgt
