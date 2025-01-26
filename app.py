from flask import Flask, request, jsonify, render_template
import os
import re
import string
import numpy as np
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")

# Initialize Flask app
app = Flask(__name__)

# Directories
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Global variables
corpus = []
document_paths = []

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r"[0-9]", "", text)  # Remove numbers
    text = re.sub(r"\s{2,}", " ", text)  # Remove extra spaces
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        print(f"Extracted text from PDF: {text[:500]}")  # Log first 500 characters
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        print(f"Extracted text from DOCX: {text[:500]}")  # Log first 500 characters
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


# Route: Render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route: Upload files
@app.route("/upload", methods=["POST"])
def upload_files():
    global corpus, document_paths
    files = request.files.getlist("file")

    for file in files:
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extract and preprocess text
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {filename}"}), 400

        if text.strip():  # Ensure the extracted text is not empty
            preprocessed_text = preprocess_text(text)
            corpus.append(preprocessed_text)
            document_paths.append(file_path)
        else:
            print(f"Failed to extract text from file: {filename}")

    print(f"Processed Corpus: {corpus}")  # Debugging log

    # Fit the TF-IDF vectorizer after processing files
    if corpus:
        fit_vectorizer()
    else:
        print("Corpus is empty. No text extracted from uploaded files.")

    return jsonify({"message": "Files uploaded and processed successfully!"}), 200

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
def fit_vectorizer():
    global vectorizer, corpus
    if corpus:
        vectorizer.fit(corpus)
        print("TF-IDF vectorizer fitted with corpus.")
    else:
        print("Corpus is empty. TF-IDF vectorizer not fitted.")


# Route: Search
@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    top_n = max(1, int(request.json.get("top_n", 5)))  # Ensure top_n is at least 1

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if not corpus or not vectorizer:
        return jsonify({"error": "No documents uploaded or vectorizer not fitted"}), 400

    # Preprocess the query
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query]).toarray()

    results = []

    # Search for the query in each document (sentence-level search)
    for doc_idx, doc in enumerate(corpus):
        sentences = nltk.sent_tokenize(doc)  # Split document into sentences
        sentences_clean = [preprocess_text(sentence) for sentence in sentences]
        sentence_vectors = vectorizer.transform(sentences_clean).toarray()

        for sent_idx, sentence_vec in enumerate(sentence_vectors):
            similarity = cosine_similarity(query_vec, sentence_vec.reshape(1, -1))[0][0]
            results.append({
                "document": document_paths[doc_idx],
                "sentence": sentences[sent_idx],
                "similarity": similarity
            })

    # Sort results by similarity score in descending order
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # Return the top N results
    return jsonify(results[:top_n]), 200

# Run Flask app
if __name__ == "__main__":
    fit_vectorizer()  # Fit the vectorizer after uploading files
    app.run(debug=True)
