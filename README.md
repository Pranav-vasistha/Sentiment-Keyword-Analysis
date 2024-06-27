# Sentiment Analysis using BERT Embeddings

This repository contains a Python script to perform sentiment analysis on text data from an Excel file using BERT embeddings. The script utilizes the DistilBERT model from the transformers library to generate embeddings and calculate sentiment scores based on user-provided phrases.

# Features

- Loads text data from an Excel file.
- Initializes DistilBERT model and tokenizer for text processing.
- Calculates BERT embeddings for both text entries and user-provided phrases.
- Computes sentiment scores based on cosine similarity between embeddings.
- Outputs average sentiment scores for each user-provided phrase.

# Requirements
Python 3.6+ ; pandas ; torch ; transformers ; openpyxl



# Run the script:
Ensure your Excel file (Final product reviews 200.xlsx) is located in the same directory as the script or provide the correct path.

python sentiment_analysis.py

When prompted, enter the phrases you want to analyze. Type 'All done' when you have finished entering phrases.

