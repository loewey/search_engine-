#
#
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import spacy
# from bs4 import BeautifulSoup
# import regex as re
# from nltk.stem import WordNetLemmatizer
# import os
#
#
# # nltk.download('stopwords')
# # nltk.download('punkt')
# # nltk.download('wordnet')
# nlp = spacy.load('en_core_web_sm')
#
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
#
#
# def simpleTokenizer(text):
#     custom_pattern = r'@\w+|\b[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+\b|[.,!?;])'
#     return re.findall(custom_pattern, text)
#
# def apply_ner(text):
#     doc = nlp(text)
#     entities = [ent.text for ent in doc.ents]
#     return entities
#
# def lemmatize_tokens(tokens):
#     return(lemmatizer.lemmatize(tokens) for token in tokens)
#
# def remove_stopwords(tokens):
#     return [token for token in tokens if token.lower() not in stop_words]
#
# def further_tokenizer(text):
#     tokens = simpleTokenizer(text)
#
#     entities = apply_ner(tokens)
#
#
#     for entity in entities:
#         tokens = [token if token.lower() != entity.lower() else f"<ENTITY : {entity}>" for token in tokens]
#
#
#     tokens = remove_stopwords(tokens)
#
#     tokens = lemmatize_tokens(tokens)
#
#     return tokens
#
# def read_from_file(directory_path):
#     results = {}
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.html'):
#             filename = os.path.join(directory_path, filename)
#             with open(filename, 'r',encoding = 'utf-8') as file:
#                 soup = BeautifulSoup(file, 'html.parser')
#                 text = soup.get_text()
#                 tokens = further_tokenizer(text)
#                 results[filename] = tokens
#     return results
#
# directory_path = 'C:\\Users\\grand\\PycharmProjects\\Search_engine\\videogames'
#
# processed_file = read_from_file(directory_path)
#
# for file,tokens in processed_file.items():
#     print(f"file : {file}")
#     print(tokens)
#     print()
import math
import os
import nltk
import re
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spacy.matcher.dependencymatcher import defaultdict

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
    # Pattern for handling words, hashtags, mentions, and game titles
    pattern = r"@\w+|#\w+|\b[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*\b|[.,!?;]"
    return re.findall(pattern, text)


# Named Entity Recognition (NER) to capture specific entities like game titles, companies, etc.
def apply_ner(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]  # Extract the entities (game titles, publishers)
    return entities


# Lemmatization function
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


# Stop word removal
def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stop_words]


# Main Tokenizer Function that combines all the steps
def advanced_tokenizer(text):
    # Step 1: Initial Tokenization (word + custom tokens)
    tokens = custom_tokenizer(text)

    # Step 2: Named Entity Recognition (NER)
    entities = apply_ner(text)

    # Replace entities with their own token for easier matching
    for entity in entities:
        tokens = [token if token.lower() != entity.lower() else f"<ENTITY:{entity}>" for token in tokens]

    # Step 3: Remove Stop Words
    tokens = remove_stopwords(tokens)

    # Step 4: Lemmatization
    tokens = lemmatize_tokens(tokens)

    return tokens


# Function to read and process all HTML files in a directory
def process_html_directory(directory_path):
    results = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text()
                tokens = advanced_tokenizer(text)
                results[filename] = tokens
    return results

def calculate_tf(tokens):
    tf = defaultdict(int)
    for token in tokens:
        tf[token] +=1
    total_tokens = len(tokens)
    return {token : count / total_tokens for token, count in tf.items()}

def calculate_idf(documents):
    total_docs = len(documents)
    doc_freq = defaultdict(int)
    for tokens in documents.values():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    return {token : math.log(total_docs/freq) for token, freq in doc_freq.items()}

def calculate_tf_idf(documents,idf):
    tf_idf = {}
    for doc, tokens in documents.items():
        tf = calculate_tf(tokens)
        tf_idf[doc] = {token: tf[token] * idf[token] for token in tf}
    return tf_idf

def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[token] * vector2.get(token,0) for token in vector1)
    norm1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

# def search_tokens(query, processed_results):
#     query_tokens = query.split()
#     results = {}
#     for file, tokens in processed_results.items():
#         matched_tokens = [token for token in tokens if any(q.lower() in token.lower() for q in query_tokens)]
#         if matched_tokens:
#             results[file] = matched_tokens
#     return results

def search_tokens(query,tf_idf,idf):
    query_tokens = advanced_tokenizer(query)
    query_tf = calculate_tf(query_tokens)
    query_vector = {token: query_tf[token] * idf.get(token, 0) for token in query_tf}

    results = []
    for doc, doc_vector in tf_idf.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        results.append((doc, similarity))

    results = sorted(results, key=lambda x: x[1], reverse=True)  # Sort by similarity
    return results

def save_results_to_file(results, output_file):
    with open(output_file, 'w') as file:
        for doc, score in results:
            file.write(f"{doc}: {score:.4f}\n")

def run_search_cli():
    directory_path = input('Enter directory path: ').strip()
    print('processing files...')
    processed_results = process_html_directory(directory_path)
    idf = calculate_idf(processed_results)
    tf_idf = calculate_tf_idf(processed_results,idf)
    print('Files processed successfully. You can now run search queries')

    while True:
        query = input('Enter search query,(or type exit to quit): ').strip()
        if query.lower() == 'exit':
            print('Exiting the program')
            break
        search_results = search_tokens(query, tf_idf, idf)
        if search_results:
            # for file, matches in search_results.items():
            #     print(f"\nfile: {file}")
            #     print("Matches:",", ".join(matches))
            print("\n Ranked Results")
            for doc, score in search_results:
                print(f"{doc}: {score:.4f}")
            save_results_to_file(search_results,"search_results.txt")
            print(f"\nResults saves to 'search_results.txt'")
        else:
            print("No matches found.")

if __name__ == '__main__':
    run_search_cli()