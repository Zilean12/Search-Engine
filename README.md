# Simple Search Engine

This is a simple search engine application built with Flask. It uses TF-IDF and cosine similarity to retrieve and rank documents based on user queries. The application preprocesses text data, builds an inverted index, calculates TF-IDF scores, and returns documents ranked by relevance.

## Features
- **Text Preprocessing**: Tokenization, stop word removal, and stemming
- **Inverted Index Construction**: Allows efficient term-based lookups
- **TF-IDF Calculation**: Measures the importance of terms in each document
- **Cosine Similarity**: Computes similarity between the query and documents for ranking
- **Web Interface**: Search through documents using a simple HTML form

## Prerequisites
- Python 3.10+
- Internet connection (for downloading NLTK stopwords)

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Zilean12/Search-Engine.git
   cd Search-Engine
2. **Install Required Packages** Install the necessary Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
3. **Download NLTK Data** Download the stopwords dataset from NLTK:

4. **Run the Application** Start the Flask app by running:
   ```bash
   python app.py

 The app will be available at `http://127.0.0.1:5000`.

Project Structure
-----------------

-   `app.py`: Main application file with text processing, TF-IDF calculation, and Flask routes.
-   `templates/index.html`: HTML template for the search interface.
-   `static/style.css`: CSS file for styling the web interface.
-   `requirements.txt`: List of required Python packages.

  Usage
-----

1.  Open the app in your browser (`http://127.0.0.1:5000`).
2.  Enter a search query in the input box and click "Search."
3.  The application will display documents ranked by relevance to the query, showing their cosine similarity scores.

Key Components
--------------

### Text Preprocessing

The text is converted to lowercase, punctuation is removed, stop words are removed, and remaining words are stemmed.

### Inverted Index

An inverted index is created to store document IDs for each unique term, facilitating fast lookup of terms in documents.

### TF-IDF Calculation

The TF-IDF score is calculated for each term in each document. TF (Term Frequency) and IDF (Inverse Document Frequency) scores are used to measure term importance.

### Cosine Similarity

The similarity between the query and each document is calculated using cosine similarity, which helps rank documents based on relevance.

Dependencies
------------

-   **Flask**: Web framework for Python
-   **NLTK**: Natural Language Toolkit for text preprocessing
-   **NumPy**: Numerical operations and vector calculations
-   **Tabulate**: Formatting data in tables for better readability in the console
-   **Colorama**: Cross-platform library for color formatting in the terminal

