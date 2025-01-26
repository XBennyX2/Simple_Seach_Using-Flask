from flask import Flask, request, jsonify, render_template
import os
import re
import string
import numpy as np
import pandas as pd
import fitz  
import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

corpus = []
document_paths = []

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text) 
    text = re.sub(r"@\w+", "", text)  
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  
    text = re.sub(r"[0-9]", "", text)  
    text = re.sub(r"\s{2,}", " ", text)  
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
        print(f"Extracted text from PDF: {text[:500]}")  
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        print(f"Extracted text from DOCX: {text[:500]}")  
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


@app.route('/')
def index():
    return render_template('index.html')

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

        
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {filename}"}), 400

        if text.strip():  
            preprocessed_text = preprocess_text(text)
            corpus.append(preprocessed_text)
            document_paths.append(file_path)
        else:
            print(f"Failed to extract text from file: {filename}")

    print(f"Processed Corpus: {corpus}")  

    if corpus:
        fit_vectorizer()
    else:
        print("Corpus is empty. No text extracted from uploaded files.")

    return jsonify({"message": "Files uploaded and processed successfully!"}), 200

vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
def fit_vectorizer():
    global vectorizer, corpus
    if corpus:
        vectorizer.fit(corpus)
        print("TF-IDF vectorizer fitted with corpus.")
    else:
        print("Corpus is empty. TF-IDF vectorizer not fitted.") # problem 1


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get("query", "").strip()
    top_n = int(request.json.get("top_n", 5))

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    processed_query = preprocess_text(query)
    query_words = set(processed_query.split())  

    results = []
    word_window = 5  

    
    for doc_idx, doc in enumerate(corpus):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', doc)
        sentences_clean = [preprocess_text(sentence) for sentence in sentences]

        for original_sentence, clean_sentence in zip(sentences, sentences_clean):
            clean_words = clean_sentence.split()
            original_words = original_sentence.split()

            for i, word in enumerate(clean_words):
                if word in query_words:
                    start_idx = max(0, i - word_window)
                    end_idx = min(len(original_words), i + word_window + 1)

                    text_portion = " ".join(original_words[start_idx:end_idx])

                    similarity = cosine_similarity(
                        vectorizer.transform([processed_query]).toarray(),
                        vectorizer.transform([" ".join(clean_words[start_idx:end_idx])]).toarray()
                    )[0][0]

                    results.append({
                        "sentence": text_portion.strip(),  
                        "similarity": similarity
                    })

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    
    limited_results = results[:top_n]

    print(f"Final Results: {limited_results}") # returns success in the back

    if not limited_results:
        return jsonify({"message": "No matching portions found."}), 200

    return jsonify(limited_results), 200


if __name__ == "__main__":
    fit_vectorizer()  
    app.run(debug=True)
