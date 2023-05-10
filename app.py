import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from transformers import pipeline
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template , request
import os
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

if not os.path.exists('uploads'):
    os.makedirs('uploads')

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'uploads'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        

        # Save the uploaded file to the server
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        with open(os.path.join(UPLOAD_FOLDER, filename), 'r', encoding='ISO-8859-1') as file:
            text = file.read()
        # Define the NLTK objects
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # # Remove HTML tags
        # text = re.sub('<[^<]+?>', '', text)
        # # Remove special characters
        # text = re.sub('[^a-zA-Z0-9 \n\.]', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)
        # # Remove stop words
        # tokens = [token for token in tokens if token not in stop_words]
        # # Lemmatize the tokens
        # tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join the tokens back into a string
        text = ' '.join(tokens)

        # Generate summary using BART model
        summarizer = pipeline("summarization", model="Vailla-Rohit/bart-base-finetuned-samsum")
        summary = summarizer(text, max_length=124, min_length=10, do_sample=False)[0]['summary_text']

        # Tokenize the summary into sentences
        sentences = sent_tokenize(summary)

        # # Create a similarity matrix using cosine similarity
        # vectorizer = TfidfVectorizer()
        # vectors = vectorizer.fit_transform(sentences)
        # similarity_matrix = cosine_similarity(vectors)

        # # Convert the similarity matrix to a graph using NetworkX
        # graph = nx.from_numpy_array(similarity_matrix)

        # # Apply the PageRank algorithm to get the sentence scores
        # scores = nx.pagerank(graph)

        # # Sort the sentences based on their scores to get the summary
        # sorted_sentences = sorted(scores, key=lambda x: sentences.index(x) if x in sentences else -1)
        # num_sentences = len(sorted_sentences)
        # summary_sentences = sorted_sentences[:num_sentences]
        # final_summary = ' '.join([sentences[idx] for idx in summary_sentences])

   

        return render_template('home.html', result=summary)
        
        

    # If the request method is GET, render the HTML page without result
    return render_template('home.html', result='')


if __name__ == '__main__':
    app.run(debug=True)
