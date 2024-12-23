import os
import re
import math
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize SpaCy for NER and WordNet for Lemmatization
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Custom Tokenizer for Hashtags, Mentions, and Game Titles (Multi-Word Entities)
def custom_tokenizer(text):
    pattern = r"@\w+|#\w+|\b[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*\b|[.,!?;]"
    return re.findall(pattern, text)


# Lemmatization function
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]


# Stop word removal
def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stop_words]


# Tokenizer pipeline
def advanced_tokenizer(text):
    tokens = custom_tokenizer(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens


# Function to read and process all HTML files in a directory
def process_html_directory(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text()
                tokens = advanced_tokenizer(text)
                documents[filename] = tokens
    return documents


# Calculate term frequency (TF)
def calculate_tf(tokens):
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    total_tokens = len(tokens)
    return {token: count / total_tokens for token, count in tf.items()}


# Calculate inverse document frequency (IDF)
def calculate_idf(documents):
    total_docs = len(documents)
    doc_freq = defaultdict(int)
    for tokens in documents.values():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    return {token: math.log((1 + total_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}


# Calculate TF-IDF for all documents
def calculate_tf_idf(documents, idf):
    tf_idf = {}
    for doc, tokens in documents.items():
        tf = calculate_tf(tokens)
        tf_idf[doc] = {token: tf[token] * idf[token] for token in tf}
    return tf_idf


# Cosine similarity between query and document vectors
def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[token] * vec2.get(token, 0) for token in vec1)
    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0


# Search for query using VSM
def search_query(query, tf_idf, idf, documents):
    query_tokens = advanced_tokenizer(query)
    query_tf = calculate_tf(query_tokens)
    query_vector = {token: query_tf[token] * idf.get(token, 0) for token in query_tf}

    results = []
    for doc, doc_vector in tf_idf.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        matched_tokens = [token for token in query_tokens if token in documents[doc]]
        results.append((doc, similarity, matched_tokens))

    results = sorted(results, key=lambda x: x[1], reverse=True)  # Sort by similarity
    return results


# Save results to a file
def save_results_to_file(results, output_file):
    with open(output_file, 'w') as file:
        for doc, score, matches in results:
            file.write(f"{doc}: {score:.4f}, Matches: {', '.join(matches)}\n")


# CLI for the search
def run_search_cli():
    directory_path = input("Enter the path to the directory containing HTML files: ").strip()
    print("Processing files...")
    documents = process_html_directory(directory_path)
    idf = calculate_idf(documents)
    tf_idf = calculate_tf_idf(documents, idf)
    print("Files processed successfully. You can now run search queries.")

    while True:
        query = input("\nEnter a search query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        results = search_query(query, tf_idf, idf, documents)
        if results:
            print("\nRanked Results:")
            for doc, score, matches in results[:10]:
                print(f"{doc}: {score:.4f}, Matches: {', '.join(matches)}")
            save_results_to_file(results, "search_results.txt")
            print("\nResults saved to 'search_results.txt'.")
        else:
            print("No matches found.")


# Run the CLI
if __name__ == "__main__":
    run_search_cli()
