Contains the datasets used:
* in original form: sts-train.csv as well as the three WMT folders, the mlqe dataset and the original datasets for the adversarial experiment (paws,freitag).
* and a slightly preprocessed form which only contains the sentences and the Semantic scores and also unifies the respective WMT sets into one file each: sts_\*.tsv, mlqe_\*.tsv and wmt_\*_full.tsv. The preprocessed datasets for the adversarial experiments are in the `adversarial` directory.

We provide preprocessed datasets, but you can recreate them yourself.
To preprocess the original dataset for our experiments run one of the following scripts depending on the dataset:

```
python preprocess_mlqe.py
--lang TEXT             The language of the dataset. de or zh
```

```
python preprocess_wmt.py
--lang TEXT             The language of the dataset. de, zh (crosslingual) or en (monolingual)
```

```
python preprocess_freitag.py
```

```
python preprocess_mlqe.py
```
