These files contain the scores for the regression.
Their files are named like `out_DATASET_LANG.TSV`

Each file has the following columns depending on the language and the dataset:

WMT/STS en:
```
sentence1 sentence2 sem syn morph bleu fasttext bert laser mUSE mover sbert sbert_wk labse mSBERT bleurt
```

WMT/STS de:
```
sentence1 sentence2 sem syn morph bleu-w bleu-s fasttext laser mUSE xmover labse mSBERT
```

WMT/STS zh:
```
sentence1 sentence2 sem syn bleu-w bleu-s fasttext laser mUSE xmover labse mSBERT
```

PAWS/Freitag:
```
label sentence1 sentence2 bert laser mUSE mover sbert sbert_wk labse mSBERT bleurt
```
* label is 1 for a paraphrase or 0 otherwise
