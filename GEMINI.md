# Gemini CLI Context - Automatic Fake News Detection

This project is a legacy college project focused on detecting fake news using various machine learning models and datasets (COVID, FakeNewsNet, ISOT, Fake News Costa Rica News, LIAR).

## Project Structure

- `code/`: Main pipeline scripts for data extraction, splitting, statistics, and model training.
- `code/special_code/`: Dataset-specific variations of the main pipeline.
- `README.md`: Project overview (to be updated).

## Tech Stack

- Python 3
- Pandas, NumPy, Scikit-learn
- NLTK (Stopwords)
- BERT (Transformers), FastText, Word2Vec, GloVe

## Guidelines for this Project

- **Refactoring**: Improve code readability, add docstrings (Google style), and use `if __name__ == "__main__":` blocks.
- **Paths**: Use relative paths where possible or provide a way to configure the base data directory.
- **Documentation**: Avoid emojis in `README.md`.
- **Maintenance**: This is a legacy project, so preserve the original logic while improving the implementation quality.
