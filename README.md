# HelvetiaDial
HelvetiaDial is a classifier for the Swiss German dialects spoken in: Lucerne, Basel, Zurich and Bern. 
It is trained and tested with the dataset of the 'German Dialects Identification' task of the VarDial 2019 workshop.
Backround information on this project can be found in the report.pdf

## Setup
To run HelvetiaDial one shound create a Python 3.10 environment (We only tested the code on Python 3.10).
Afterward the requirements in requirements.txt should be installed using pip. 
The git repo contains the embeddings. If one wants to create them one needs to:
- Dowload the GDI-2019 dataset from https://drive.switch.ch/index.php/s/DZycFA9DPC8FgD9 
- In create_embeddings.ipynb the path to the dataset needs to be set to the unzipped folder.
- Run the jupyter notebook 'create_embeddings.ipynb'. Takes approximately 1 hour.
## Files Overview
Following table gives an overview of the files in this repository.

| File | Description                                     |
| --- |-------------------------------------------------|
| bert_swiss_lm.py | Code to create BERT embeddings                  |
| byte_pair_tfidf_vectorizer.py | Code to create Byte Pair TF-IDF embeddings      |
| create_embeddings.ipynb | Notebook to create the embeddings               |
| experimental_predictors.py | Multiple classifiers, that we experimentet with |
| experiments.py | Code to create the plots of the report          |
| gdi_loader.py | Code to load the GDI-2019 dataset               |
| helper_functions.py | helper functions                                |
| helvetia_dial.py | Final architecture (run to calculate f1-score)  |
| README.md | This file                                       |
| more_experiments.ipynb | More experiments                                |
| report.pdf | Report on the project                           |
| requirements.txt | Requirements to run the code                    |
