# IMDB Review Sentiment Analysis

This project explores classical NLP pipelines for predicting the sentiment of IMDB movie reviews. The goal is to classify reviews as *positive* or *negative* based on their text,. The dataset (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) is balanced for sentiment (positive/negative) and contains 5000 reviews (a subsample was chosen).

### Pipeline Steps
1. **Preprocessing**: Tokenization (regex or NLTK), lowercasing, stopword removal, stemming/lemmatization.
2. **Feature Extraction**: TF-IDF vectorization (unigrams/bigrams).
3. **Classification**: Logistic Regression and MLP.
4. **Evaluation**: 5-fold stratified cross-validation, reporting accuracy, F1, ROC-AUC, and PR-AUC.

## Preprocessing Configurations
Multiple preprocessing configurations were tested, including:
- Regex vs. NLTK tokenization
- With/without stopword removal
- With/without stemming or lemmatization

## Results Summary
- **Logistic Regression** and **MLP** both performed well, with LR slightly outperforming Logistic Regression in most configurations.
- Removing stopwords or applying stemming/lemmatization did not always improve results, suggesting that some stopwords or word forms may carry useful information for this task.