Contains data needed for the calculation of morphological scores.
* \*_plaintext_\*.tsv are plaintext versions of the datasets with one sentence per line.
* /lexicons contains the morphological word-pair lexicons. 
* /embeddings contains the embeddings used: 
  * /fasttext contains the embeddings which you have generated with `embed_text.py`
  * /retrofitted contains the embeddings produced be using https://github.com/sarnthil/retrofitting to retrofit those embeddings with the lexicons.

Run 
```
python to_plaintext.py [OPTIONS]

OPTIONS:
--input TEXT         Dataset as tsv file from data/raw
--output TEXT
```
to convert a dataset to plaintext for generating the embeddings.
