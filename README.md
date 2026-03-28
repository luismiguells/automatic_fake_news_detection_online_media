# Automatic Fake News Detection in Online Media

This repository contains the code for automatic fake news detection using various machine learning models and datasets. This project was developed as part of a college thesis.

## Project Overview

The goal of this project is to identify fake news by leveraging multiple text representation techniques and classification models. The pipeline is organized into clear stages to facilitate experimentation and reproducibility.

## Datasets

The project uses the following datasets:
- COVID Dataset
- FakeNewsNet (FNN) Dataset
- ISOT Dataset
- Fake News Costa Rica News (FNCN / FNCS) Dataset
- LIAR Dataset

Note: The datasets are not included in this repository. Please contact the author for access.

## Project Structure

The code is organized into a tiered pipeline within the `code/` directory. Each folder represents a step in the process:

### 1. Data Preparation (`code/data_preparation/`)
- `extraction.py`: Extract and clean raw data from various sources (Excel, CSV, TSV).
- `splitting.py`: Split the processed data into training, validation, and test sets.

### 2. Analysis (`code/analysis/`)
- `statistics.py`: Calculate dataset statistics such as word counts, vocabulary size, and label distributions.

### 3. Models (`code/models/`)
This directory contains various model implementations using different text representations:
- `tfidf.py`: Baseline models using TF-IDF.
- `glove.py`, `w2v.py`, `fasttext.py`: Models using pre-trained word embeddings.
- `lsa_nmf.py`: Topic modeling based approaches (SVD/NMF).
- `nn.py`, `nn_glove.py`, etc.: Neural Network architectures.
- `bert.py`: Transformer-based models using BERT.

### 4. Special Experiments (`code/special_experiments/`)
Tailored experiments for specific datasets and advanced techniques:
- `fncs/`: Scripts optimized for the Costa Rica News dataset.
- `liar/`: Scripts optimized for the LIAR dataset.
- `bert/`: Advanced BERT experiments, including bagging and specialized configurations.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Transformers (for BERT models)
- FastText / Gensim (for word embeddings)

## Usage

Follow the numerical order of the directories to run the full pipeline:

1.  **Prepare Data**:
    ```bash
    python code/data_preparation/extraction.py
    python code/data_preparation/splitting.py
    ```
2.  **Analyze Data**:
    ```bash
    python code/analysis/statistics.py
    ```
3.  **Train Models**:
    ```bash
    python code/3_models/tfidf.py
    ```

## Authors

Luis Miguel López Santamaría - [luis_miguel@outlook.com]
