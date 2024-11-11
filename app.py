import re
import string
import numpy as np
from collections import defaultdict
from math import log
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request
from tabulate import tabulate
from colorama import Fore, Style

# Initialize the Flask app
app = Flask(__name__)

# Example documents
documents = {
    "doc1": "Climate change is one of the most pressing issues of our time. Global warming impacts weather patterns.",
    "doc2": "Artificial intelligence is rapidly changing industries, with applications in healthcare, finance, and beyond.",
    "doc3": "Quantum computing promises to revolutionize the field of computing by solving complex problems much faster.",
    "doc4": "Sustainability efforts are vital to combating climate change. Renewable energy sources are part of the solution.",
    "doc5": "AI can assist in predicting medical conditions based on patient data and improve healthcare outcomes."
}

# Initialize stop words and stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# # Preprocessing function
# def preprocess(text):
#     print("\nOriginal Text:", text)
#     text = text.lower()
#     text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
#     tokens = text.split()  # Tokenize
#     processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]  # Remove stop words and apply stemming
#     print("After Stop Words Removal and Stemming:", processed_tokens)
#     return processed_tokens

# # Step 1: Preprocess all documents
# preprocessed_docs = {doc_id: preprocess(text) for doc_id, text in documents.items()}

# Preprocessing function with colored output
def preprocess(text):
    # Display the original text with color formatting
    print(f"\n{Fore.BLUE}Original Text:{Style.RESET_ALL} {Fore.CYAN}{text}{Style.RESET_ALL}")
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize
    
    # Remove stop words and apply stemming
    processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    # Display the processed tokens with color formatting
    formatted_tokens = ', '.join([f"{Fore.YELLOW}{token}{Style.RESET_ALL}" for token in processed_tokens])
    print(f"{Fore.GREEN}After Stop Words Removal and Stemming:{Style.RESET_ALL} [{formatted_tokens}]")
    
    return processed_tokens

#  Step 1: Preprocess all documents
preprocessed_docs = {doc_id: preprocess(text) for doc_id, text in documents.items()}

# Step 2: Building the Inverted Index
inverted_index = defaultdict(list)
for doc_id, tokens in preprocessed_docs.items():
    for token in tokens:
        if doc_id not in inverted_index[token]:
            inverted_index[token].append(doc_id)

# Display Inverted Index with color and tabular format
def display_inverted_index(index):
    print("\nInverted Index:\n")
    headers = ["Term", "Document IDs"]
    table = [[f"{Fore.CYAN}{term}{Style.RESET_ALL}", f"{Fore.GREEN}{', '.join(doc_ids)}{Style.RESET_ALL}"] for term, doc_ids in index.items()]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

display_inverted_index(inverted_index)

# Step 3: TF-IDF Calculation
def compute_tf_idf(docs, index):
    N = len(docs)  # Number of documents
    tf_idf = {doc_id: {} for doc_id in docs}
    for term, doc_ids in index.items():
        idf = log(N / len(doc_ids))  # IDF calculation
        for doc_id in doc_ids:
            tf = docs[doc_id].count(term) / len(docs[doc_id])  # TF calculation
            tf_idf[doc_id][term] = tf * idf
    return tf_idf

# Display TF-IDF in a color-coded table format
def display_tf_idf(tf_idf):
    headers = [f"{Fore.BLUE}Document{Style.RESET_ALL}", 
               f"{Fore.MAGENTA}Term{Style.RESET_ALL}", 
               f"{Fore.GREEN}TF-IDF Score{Style.RESET_ALL}"]
    table = []

    for doc_id, term_scores in tf_idf.items():
        for term, score in term_scores.items():
            # Adding color to each column
            colored_doc_id = f"{Fore.CYAN}{doc_id}{Style.RESET_ALL}"
            colored_term = f"{Fore.YELLOW}{term}{Style.RESET_ALL}"
            colored_score = f"{Fore.GREEN}{round(score, 4)}{Style.RESET_ALL}"
            table.append([colored_doc_id, colored_term, colored_score])

    print("\nTF-IDF Scores Table:")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

# Calculate and display TF-IDF for documents
tf_idf = compute_tf_idf(preprocessed_docs, inverted_index)
display_tf_idf(tf_idf)

# Step 4: Cosine Similarity
def vectorize(tf_idf, query_terms):
    terms = list(set(query_terms + [term for doc in tf_idf.values() for term in doc]))
    vectors = {doc_id: [doc.get(term, 0) for term in terms] for doc_id, doc in tf_idf.items()}
    query_vector = [1 if term in query_terms else 0 for term in terms]
    return vectors, query_vector

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

# Flask route to display the search form and results
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_text = ""
    if request.method == "POST":
        query_text = request.form.get("query")
        query_tokens = preprocess(query_text)
        
        print("\nProcessed Query Tokens:", query_tokens)
        
        # Compute vectors and cosine similarity
        vectors, query_vector = vectorize(tf_idf, query_tokens)
        
        # Print cosine similarity for each document
        scores = {doc_id: cosine_similarity(query_vector, vector) for doc_id, vector in vectors.items()}
        print("\nCosine Similarity Scores:")
        for doc_id, score in scores.items():
            print(f"{doc_id}: {score}")
        
        # Sort documents based on cosine similarity scores
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results = [(doc_id, documents[doc_id], score) for doc_id, score in sorted_scores if score > 0]

    return render_template("index.html", results=results, query_text=query_text)

if __name__ == "__main__":
    app.run(debug=True)
