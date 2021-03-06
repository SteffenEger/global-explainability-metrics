# Global Explainability of BERT-Based Evaluation Metrics by Disentangling along Linguistic Factors

Code and data for our [EMNLP 2021](https://arxiv.org/abs/2110.04399) paper. We represent metrics as linear combinations of linguistic factors, thus contribute to their global explainability.

# Dependencies
## For calculating scores:
* Python 3.6
* [XMoverScore](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/), expected in folder xmover
* [MoverScore](https://github.com/AIPHES/emnlp19-moverscore), expected in folder mover
* [SBERT-WK](https://github.com/BinWang28/SBERT-WK-Sentence-Embedding), expected in folder sbert_wk
* [laserembeddings](https://github.com/yannvgn/laserembeddings) for LASER
* [bleurt-base-128](https://github.com/google-research/bleurt) checkpoint for BLEURT, expected in folder bleurt-base-128
* StanfordDependencyParser files from https://nlp.stanford.edu/software/lex-parser.shtml **calcuations were done with version 3.5.2 (select from release history)**
  * The zip file should be extracted such that the jar files lie inside the folder "stanford-parser-full-2015-04-20". (alternatively the path can be changed by editing lines 57/58 in score.py.
## Other depencies
* A conda environment with all necessary dependencies is stored at `environment.yml`

# Usage
We provide most of the necessary files, but you can also recreate them yourself with our scripts.
You can directly calculate  the regression with our scored files, or you can calculate them yourself.
## Calculating Scores

To obtain scores for all sentence pairs in an input file run:
```
python score.py [OPTIONS]

Options:
--input TEXT                Path to input file [required]
--output TEXT               Path to output file [required]  
--lang TEXT                 Language of source sentences, one of 'en', 'de' or 'zh' [required, default:'en']                 
--embeddings TEXT           Path to file with morphological embeddings [only used if lang != 'zh']
--translations TEXT         Path to file which contains the translations for the lexical overlap in the cross-lingual case [required if lang != en]

```
* The **input** file needs to be a tsv (i.e. tab-delimited csv) file where each line/row has the form "Sem_score  Sent1  Sent2". Such files for the datasets used are provided in /data/raw.
* This will **output** another tsv file with a sentence pair and its scores in a single line
* The **embeddings** file (if used) is of the format produced by using https://github.com/sarnthil/retrofitting. Already retrofitted files are provided in /data/morph/embeddings/retrofitted.
* The **translations** file must be a pickle file. It is provided at `data/translated`

## Regression

```
python regress.py [OPTIONS]

Options:
--input_file TEXT           Path to input file [required]
--lang TEXT                 Language of source sentences, one of 'en', 'de' or 'zh' [required, default:'en']                 
```
* The **input** file needs to be of the format produced by score.py for the respective language.

* This will print out summaries for the regressions.

## Preprocessing Steps for Morphological Embeddings
### Generating Embeddings

```
python embed_text.py [OPTIONS]

Options:
--input TEXT                Path to input file [required]
--lang TEXT                 Language of source sentences, one of 'en' or 'de' [required, default:'en']                
--output TEXT               Path to output file [required]  
```
* The **input** file needs to be plain text with one sentence per line. Such files for the used datasets are provided in /data/morph.
* This **outputs** fasttext embeddings for the input file into the output file (as a tsv file of the form "Word Embedding").

### Creating a Morphological Lexicon

```
python create_lexicon.py [OPTIONS]

Options:
--input TEXT                Path to input file [required]
--lang TEXT                 Language of source sentences, one of 'en' or 'de' [required, default:'en']                
--output TEXT               Path to output file [required]  
```
* As above, the **input** file needs to be plain text with one sentence per line. Such files for the used datasets are provided in /data/morph.
* This **outputs** a tsv file with one pair of words that is morphologically close per line.

### Retrofitting
Clone the git repository from https://github.com/sarnthil/retrofitting and retrofit the embeddings generated by `embed_text.py`. Run 
```
python retrofit.py  [OPTIONS]

Options:
--input TEXT
--lexicon TEXT 
--output TEXT`
```
* The **input** file is an embedding generated by `embed_text.py`
* The **lexicon** file is a lexicon generated by `create_lexicon.py`


## Translating STS Dataset
```
python translate_sts.py [OPTIONS]

Options:
--input TEXT                Path to input file [required, default:'./data/raw/sts-train.csv']
--lang TEXT                 Target language, one of 'de' or 'zh' (others are possible, see https://pypi.org/project/googletrans/) [required, default:'zh']                
--output TEXT               Path to output file [required]  
```
* The **input** file is of the format of the original files for the STS datasets (here in /data/raw/sts-train.csv).
* This **outputs** a tsv file with the semantic scores, original first sentences and translated second sentences to a tsv file of the form "Sem_Scores  Sent1 Translated_Sent2".

## Translating for cross-lingual lexical overlap scores
```
python translate_lex.py 
--input TEXT              Path to input file. [required]
--lang TEXT               Source language [required: de or zh]
--output TEXT             Path to output file [required]
```
* The **input** file must be a TSV file (score tab sentence 1 tab sentence 2). These files are provided at `data/raw`

## Adversarial experiment
### Create the scores
You can create the scores yourself or use the provided scores at `data/scored/adversarial`.
```
python score_adversarial.py [OPTIONS]

Options:
--input TEXT                Path to input file [required]
--output TEXT               Path to output file [required]  
```
* The **input** file needs to be a tsv (i.e. tab-delimited csv) file where each line/row has the form "Sem_score  Sent1  Sent2". Such files for the datasets used are provided in /d
* This will **output** another tsv file with a sentence pair and its scores in a single line

### Calculate the lexical overlap
```
python adversarial_lex_overlap.py [OPTIONS]

Options:
--input TEXT                Path to score file generated byscore_adversarial.py [required]
```
* We provide **input** files: `data/score/out_freitag.tsv`, `data/score/out_paws.tsv`

## Ensembling of models
### Calculate scores
Run `python ensembling_scores.py --corpus WMT19 --csv out.csv --metrics xms --lang de` for the scores of the reference-free models

* `--corpus` The dataset. WMT17, WMT18 or WMT19
* `--csv` The csv file where the scores (for each system, sentence, language pair) will be saved
* `--metrics` List of metrics (separated by comma) for which scores shall be calculated. If not given, every metric will be calculated.
* `--lang` List of languages (separated by comma) for which scores shall be calculated. If not given, every language pair will be calculated. 

Run `python ensembling_scores_reference_based.py` with the same arguments for the reference-based models.

### Evaluate scores
To evaluate the scores of the reference-free models run `python ensembling_evaluate.py --avg False --eval seg --rr True --scores out.csv --lang de --metrics xmspp`

* `--avg` If true, the metrics will be combined and averaged. Only one correlation coefficient for each language. If false, for each metric-language pair a correlation coefficient will be calculated.
* `--eval` seg or sys . Segment or system level evaluation
* `--rr` Bool. Use only for segment level evaluation. If True, the evaluation uses relative ranking. This results in a Kendall correlation. If false, th evaluation uses direct assessment.  Pearson correlation. Check which dataset provides which evaluation
* `--scores` The csv file which was generated by main.py
* `--lang` List of languages (separated by comma) which shall be evaluated. If not given, every language pair will be calculated. 
* `--metrics` List of metrics (separated by comma) which shall be evaluated. If not given, every metric will be calculated. `--avg` determines if these metrics will be combined or evaluated separately.

You can see which kind of evaluation (relative ranking or direct assessment) is available for each dataset in the section Datasets below. The script `ensembling_evaluate_reference_based.py` does  the same evaluation as above for the reference-based models.

### Combine scores
Set `--avg True` when running `ensembling_evaluate.py`. The scores of each metric which is specified at `--metrics` will be averaged for each sentence pair. This creates a new score for each sentence pair. These scores are evaluated.

### Datasets

* WMT19 System: Direct assessment (Pearson correlation). Segment: Relative ranking (Kendall correlation)
* WMT18 System: Direct assessment (Pearson correlation). Segment: Relative ranking (Kendall correlation)
* WMT17 System: Direct assessment (Pearson correlation). Segment: Direct assessment (Pearson correlation)

# Provided Files

* /data/raw contains the datasets used, both in original form (sts-train.csv as well as the three WMT folders) and a slightly preprocessed form which only contains the sentences and the Semantic scores and also unifies the respective WMT sets into one file each (sts_\*.tsv and wmt_\*_full.tsv).
* /data/morph contains data needed for the calculation of morphological scores.
  * \*_plaintext_\*.tsv are plaintext versions of the datasets with one sentence per line.
  * /data/morph/embeddings contains the embeddings used. /fasttext contains the original fasttext embeddings while /retrofitted contains the embeddings produced be using https://github.com/sarnthil/retrofitting to retrofit those embeddings with the lexicons.
* /data/scored contains the output files with all calculated scores (as output by score.py) seperated by WMT and STS and named including either 'mono' to indicate English-English sentence pairs or 'cross_\*' to indicate cross-lingual sentence pairs as well as the specific non-English language.
* /data/translated contains the translations which are needed for the cross-lingual lexical overlap scores


### Citation

```
@inproceedings{kaster-et-al-2021-global,
    title = "Global Explainability of BERT-Based Evaluation Metrics by Disentangling along Linguistic Factors",
    author = "Kaster, Marvin  and
      Zhao, Wei  and
      Eger, Steffen",
    booktitle = "EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
