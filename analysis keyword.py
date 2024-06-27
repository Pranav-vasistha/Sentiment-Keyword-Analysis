import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Load the Excel file
file_path = 'Final product reviews 200.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Initialize the DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to tokenize and get embeddings for a given text
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings

# Function to calculate sentiment score based on BERT embeddings
def calculate_sentiment_score(text_series, phrases):
    phrase_embeddings = get_bert_embeddings(phrases)
    phrase_embeddings = phrase_embeddings.repeat(len(text_series), 1)
    
    text_embeddings = get_bert_embeddings(text_series.tolist())
    
    similarity_scores = torch.cosine_similarity(text_embeddings, phrase_embeddings, dim=1)
    sentiment_scores = similarity_scores.numpy()
    
    return sentiment_scores

# Get user input for phrases
phrases = []
while True:
    phrase = input("Enter a phrase (or 'All done' to finish): ")
    if phrase.lower() == 'all done':
        break
    phrases.append(phrase)

# Extract and calculate average sentiment scores for each phrase in the 'Comments' column
print("\nAverage Sentiment Scores for Phrases:")
for phrase in phrases:
    sentiment_scores = calculate_sentiment_score(df['Comments'], phrase)
    average_score = sentiment_scores.mean()
    print(f"Phrase: '{phrase}'")
    print(f"Average Sentiment Score: {average_score:.4f}")
    print("\n")
