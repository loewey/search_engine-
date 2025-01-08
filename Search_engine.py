# import os
# import re
# import math
# import nltk
# import spacy
# import pickle
# from nltk import ngrams
# from bs4 import BeautifulSoup
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict
# from nltk.corpus import wordnet as wn
# from nltk.stem import PorterStemmer
# import time
#
#
# # Download NLTK resources
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')
#
# # Initialize SpaCy for NER and WordNet for Lemmatization
# nlp = spacy.load('en_core_web_md')
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
#
# AGE_RATINGS = ['Everyone', 'Teen', 'Mature', 'TBA', 'Everyone 10+', 'Rating Pending']
#
# #config flags ( for the testing )
#
# CONFIG = {
#     'use_stemming': False,
#     'use_ner': True,
#     'use_bigrams':True,
#     'use_metadata':True,
#     'use_lemmatization':True,
#     'use_stopword_removal':True,
#     'use_query_expansion':True
# }
#
# WEIGHTS = {
#     'title': 3,
#     'heading': 2,
#     'metadata': 2,
#     'ner': 2.5,
#     'bigrams':2,
#     'body':1
# }
#
# # Custom Tokenizer for Hashtags, Mentions, and Game Titles (Multi-Word Entities)
# def custom_tokenizer(text):
#     pattern = r"@\w+|#\w+|\b[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*\b|[.,!?;]"
#     return re.findall(pattern, text)
#
#
# # Lemmatization function
# def lemmatize_tokens(tokens):
#     if CONFIG['use_lemmatization']:
#         return [lemmatizer.lemmatize(token.lower()) for token in tokens]
#     return tokens
#
# def stem_tokens(tokens, text):
#     # Check if stemming is enabled
#     if CONFIG['use_stemming']:
#         # Extract named entities if NER is enabled
#         named_entities = []
#         if CONFIG['use_ner']:
#             named_entities = extract_named_entities(text)  # Extract named entities only if enabled
#
#         # Convert named_entities list to a set for faster lookup
#         named_entities_set = set(named_entities)
#
#         # Stem tokens but keep named entities intact
#         return [stemmer.stem(token) if token.lower() not in named_entities_set else token for token in tokens]
#
#     return tokens
#
#
# def extract_named_entities(text):
#     if CONFIG['use_ner']:
#         doc = nlp(text)
#         entities = [ent.text for ent in doc.ents]
#         print(f"Extracted named entities: {entities}")  # Debugging: Print named entities
#         return entities
#     return []
#
# def generate_bigrams(tokens,text):
#     if CONFIG['use_bigrams']:
#         bigrams_list = list(ngrams(tokens,2))
#         stemmed_bigrams = list(ngrams(stem_tokens(tokens,text),2))
#         return [' '.join(bigram) for bigram in bigrams_list + stemmed_bigrams]
#     return []
#
# # Stop word removal
# def remove_stopwords(tokens):
#     if CONFIG['use_stopword_removal']:
#         return [token for token in tokens if token.lower() not in stop_words]
#     return tokens
# def advanced_tokenizer(text):
#     tokens = custom_tokenizer(text)
#     print(f"Tokens after custom tokenizer: {tokens}")
#
#     tokens = remove_stopwords(tokens)
#     print(f"Tokens after stopword removal: {tokens}")
#
#     tokens = lemmatize_tokens(tokens)
#     print(f"Tokens after lemmatization: {tokens}")
#
#     tokens = stem_tokens(tokens, text)  # Apply stemming with named entity preservation
#     print(f"Tokens after stemming: {tokens}")
#
#     bigrams = generate_bigrams(tokens,text)
#     print(f"Bigrams: {bigrams}")
#
#     return tokens + bigrams
#
# #Function to read and process all HTML files in a directory
# def process_html_directory_including_weights(directory_path):
#     documents = {}
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.html'):
#             file_path = os.path.join(directory_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 soup = BeautifulSoup(file, 'html.parser')
#
#                 text = soup.get_text()
#
#                 named_entites = extract_named_entities(text)
#
#                 title_tokens = advanced_tokenizer(soup.title.get_text()) if soup.title else[]
#                 heading_tokens = advanced_tokenizer(' '.join([h.get_text() for h in soup.find_all(['h1', 'h2'])]))
#
#
#                 weighted_tokens = title_tokens + heading_tokens + advanced_tokenizer(text)
#                 weighted_tokens.extend(named_entites * 2)
#
#
#                 documents[filename] = weighted_tokens
#     return documents
#
# def process_html_directory(directory_path):
#     documents = {}
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.html'):
#             file_path = os.path.join(directory_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 soup = BeautifulSoup(file, 'html.parser')
#                 text = soup.get_text()
#
#                 #all the named entities
#                 named_entites = extract_named_entities(text)
#
#                 # Extract and tokenize title (weight = 3x)
#                 title_tokens = advanced_tokenizer(soup.title.get_text()) * int(WEIGHTS['title'])if soup.title else []
#
#                 # Extract and tokenize h1, h2 headings (weight = 2x)
#                 heading_tokens = advanced_tokenizer(' '.join([h.get_text() for h in soup.find_all(['h1', 'h2'])])) * int(WEIGHTS['heading'])
#
#
#                 meta_desc_tokens, meta_keywords_tokens = [], []
#                 if CONFIG['use_metadata']:
#                     # Extract and tokenize meta description (weight = 2x)
#                     meta_desc = ' '.join([meta['content'] for meta in soup.find_all('meta', {'name': 'description'})])
#                     meta_desc_tokens = advanced_tokenizer(meta_desc) * int(WEIGHTS['metadata'])
#
#                     # Extract and tokenize meta keywords (weight = 2x)
#                     meta_keywords = ' '.join([meta['content'] for meta in soup.find_all('meta', {'name': 'keywords'})])
#                     meta_keywords_tokens = advanced_tokenizer(meta_keywords) * int(WEIGHTS['metadata'])
#
#                 # Tokenize body text (default weight)
#                 body_tokens = advanced_tokenizer(text)
#                 bigram_tokens = generate_bigrams(body_tokens) * int(WEIGHTS['bigrams'])
#
#                 # Combine all weighted tokens
#                 weighted_tokens = (title_tokens + heading_tokens +
#                                    meta_desc_tokens + meta_keywords_tokens +
#                                    body_tokens + bigram_tokens)
#
#                 # Double weight for named entities
#                 weighted_tokens.extend(named_entites * int(WEIGHTS['ner']))
#
#                 # Store tokens for each document
#                 documents[filename] = weighted_tokens
#
#     return documents
#
#
#
# #Calculate term frequency (TF)
# def calculate_tf(tokens):
#     tf = defaultdict(int)
#     for token in tokens:
#         tf[token] += 1
#     total_tokens = len(tokens)
#     return {token: count / total_tokens for token, count in tf.items()}
#
#
# # Calculate inverse document frequency (IDF)
# def calculate_idf(documents):
#     total_docs = len(documents)
#     doc_freq = defaultdict(int)
#     for tokens in documents.values():
#         unique_tokens = set(tokens)
#         for token in unique_tokens:
#             doc_freq[token] += 1
#     return {token: math.log((1 + total_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}
#
#
# # Calculate TF-IDF for all documents
# def calculate_tf_idf(documents, idf):
#     tf_idf = {}
#     for doc, tokens in documents.items():
#         tf = calculate_tf(tokens)
#         tf_idf[doc] = {token: tf[token] * idf[token] for token in tf}
#     return tf_idf
#
#
# # Cosine similarity between query and document vectors
# def cosine_similarity(vec1, vec2):
#     dot_product = sum(vec1[token] * vec2.get(token, 0) for token in vec1)
#     norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
#     norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
#     return dot_product / (norm1 * norm2) if norm1 and norm2 else 0
#
#
# def expand_query(query_tokens):
#     expanded_tokens = set(query_tokens)
#     for token in query_tokens:
#         synonyms = set()
#         for syn in wn.synsets(token):
#             for lemma in syn.lemmas():
#                 synonyms.add(lemma.name())
#         expanded_tokens.update(synonyms)
#     return list(expanded_tokens)
#
# # Search for query using VSM
# def search_query(query, tf_idf, idf, documents):
#     query_tokens = advanced_tokenizer(query) #allows for multi-term query's
#     query_entities = extract_named_entities(query)
#
#    # Identify age ratings in the query
#     query_age_ratings = [token for token in query_tokens if token in AGE_RATINGS]
#
#     if CONFIG['use_query_expansion']:
#         expanded_query_tokens = expand_query(query_tokens)
#     else:
#         expanded_query_tokens = query_tokens
#     query_tf = calculate_tf(expanded_query_tokens)
#     query_vector = {token: query_tf[token] * idf.get(token, 0) for token in query_tf}
#
#     results = []
#     for doc, doc_vector in tf_idf.items():
#
#         similarity = cosine_similarity(query_vector, doc_vector)
#         if any(age in documents[doc] for age in query_age_ratings):
#             similarity+= .7 #boost for age rating
#
#         if any(ent in documents[doc] for ent in query_entities):
#             similarity+= .5
#         matched_tokens = [token for token in expanded_query_tokens if token in documents[doc]]
#         results.append((doc, similarity, matched_tokens))
#
#     results = sorted(results, key=lambda x: x[1], reverse=True)  # Sort by similarity
#     return results
#
#
# # Save results to a file
# def save_results_to_file(results, output_file):
#     with open(output_file, 'w') as file:
#         for rank,(doc, score, matches) in enumerate(results, start=1):
#             file.write(f"Rank: {rank} | {doc}: {score:.4f}, Matches: {', '.join(matches)}\n")
#
#
# # CLI for the search
# def run_search_cli():
#     print("Select mode: ")
#     print("1. normal Mode")
#     print("2. Testing mode (Choose preprocessors)")
#
#     mode = input("Enter 1 or 2: ").strip()
#
#
#     if mode == '2':
#         print('running in Testing mode')
#
#         CONFIG['use_stemming'] = input("Enable stemming? (y/n): ").strip().lower() == 'y'
#         CONFIG['use_ner'] = input('Enable NER? (y/n): ').strip().lower() == 'y'
#         CONFIG['use_bigrams'] = input("Enable bigrams? (y/n): ").strip().lower() == 'y'
#         CONFIG['use_metadata'] = input('Enable metadata? (y/n): ').strip().lower() =='y'
#         CONFIG['use_lemmatization'] = input('Enable lemmatization? (y/n): ').strip().lower() == 'y'
#         CONFIG['use_stopword_removal'] = input('Enable stopword removal? (y/n): ').strip().lower() == 'y'
#         CONFIG['use_query_expansion'] = input('Enable Query Expansion? (y/n): ').strip().lower() == 'y'
#
#     else:
#         print('running in Efficiency mode')
#
#     directory_path = input("Enter the path to the directory containing HTML files: ").strip()
#     print("Processing files...")
#
#     # documents = measure_baseline(directory_path)
#     # idf = calculate_idf(documents)
#     # tf_idf = calculate_tf_idf(documents,idf)
#
#
#     pickle_file = 'documents_cache.pkl'
#
#     if os.path.exists(pickle_file):
#         with open(pickle_file, 'rb') as file:
#             documents, idf, tf_idf = pickle.load(file)
#         print("loaded cache data")
#     else:
#         documents = process_html_directory(directory_path)
#         idf = calculate_idf(documents)
#         tf_idf = calculate_tf_idf(documents, idf)
#         with open(pickle_file, 'wb') as file:
#             pickle.dump((documents,idf , tf_idf), file)
#         print("Files processed successfully. You can now run search queries.")
#
#
#
#
#     while True:
#         query = input("\nEnter a search query (or type 'exit' to quit): ").strip()
#         if query.lower() == 'exit':
#             print("Exiting the program.")
#             break
#         results = search_query(query, tf_idf, idf, documents)
#         if results:
#             print("\nRanked Results:")
#             for rank,  (doc, score, matches) in enumerate(results[:10],start=1):
#                 file_path = os.path.join(directory_path,doc)
#                 with open(file_path,'r',encoding='utf-8') as file:
#                     soup = BeautifulSoup(file,'html.parser')
#                     content = soup.get_text()
#                     snippet = ""
#                     if matches:
#                         match = re.search(matches[0], content, re.IGNORECASE)
#                         if match:
#                             start = max(0, match.start() - 50)  # 50 characters before match
#                             end = min(len(content), match.end() + 50)  # 50 after match
#                             snippet = content[start:end].replace('\n', ' ')  # Clean up newlines
#
#                 print(f"Rank {rank}: {doc} - Score: {score:.4f}")
#                 print(f"  URL: {file_path}")
#                 print(f"  Content Snippet: {snippet}...")
#                 print(f"  Matches: {', '.join(matches)}")
#                 print("-" * 50)  # Separator line for clarity
#             save_results_to_file(results, "search_results.txt")
#             print("\nResults saved to 'search_results.txt'.")
#         else:
#             print("No matches found.")



"""
in this code i have done
- lemmanization
- stemming
- bigrams
- removal of stopwords
- NER - named entity recoginition
- meta data ranking
- term ranking - Title, header, genre ect
- tf-idf ranking
- Vector calculation for comparison of terms
- query expansion
- save results to external text file
- outputted results to user
- implemented pickles.py to load and dump objects faster

"""



# Run the CLI
# if __name__ == "__main__":
#     run_search_cli()


# import os
# import time
# import nltk
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.util import bigrams
# from bs4 import BeautifulSoup
#
# # Initialize NLTK and spaCy
# nltk.download('punkt')
# nltk.download('stopwords')
#
# # Load spaCy model for NER (ensure you have installed it: pip install spacy && python -m spacy download en_core_web_sm)
# nlp = spacy.load("en_core_web_sm")
#
# # Initialize preprocessing objects
# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))
#
# # Function to read HTML files from a directory and extract text
# def read_html_files_from_directory(directory):
#     text_data = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".html"):  # Only process HTML files
#             file_path = os.path.join(directory, filename)
#             with open(file_path, "r", encoding="utf-8") as file:
#                 html_content = file.read()
#                 soup = BeautifulSoup(html_content, "html.parser")
#                 text = soup.get_text()  # Extracts text from the HTML
#                 text_data.append(text)
#     return text_data
#
# # Function to track time for each preprocessing step (including NER and Bigram creation)
# def track_time_for_preprocessing(text_data):
#     # Initialize a variable to accumulate total time
#     total_time = 0
#
#     # Tokenization step
#     start_time = time.time()
#     tokenized_data = [word_tokenize(sentence) for sentence in text_data]
#     end_time = time.time()
#     tokenization_time = end_time - start_time
#     total_time += tokenization_time
#     print(f"Tokenization took {tokenization_time:.4f} seconds.")
#
#     # Stopwords removal step
#     start_time = time.time()
#     filtered_data = [
#         [word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_data
#     ]
#     end_time = time.time()
#     stopwords_removal_time = end_time - start_time
#     total_time += stopwords_removal_time
#     print(f"Stopwords removal took {stopwords_removal_time:.4f} seconds.")
#
#     # Named Entity Recognition (NER) step
#     start_time = time.time()
#     ner_data = []
#     for sentence in filtered_data:
#         text = ' '.join(sentence)  # Join sentence back to text for NER processing
#         doc = nlp(text)  # Apply spaCy NER
#         ner_entities = [(ent.text, ent.label_) for ent in doc.ents]
#         ner_data.append(ner_entities)
#     end_time = time.time()
#     ner_time = end_time - start_time
#     total_time += ner_time
#     print(f"NER took {ner_time:.4f} seconds.")
#
#     # Bigram creation step
#     start_time = time.time()
#     bigram_data = [list(bigrams(sentence)) for sentence in filtered_data]
#     end_time = time.time()
#     bigram_creation_time = end_time - start_time
#     total_time += bigram_creation_time
#     print(f"Bigram creation took {bigram_creation_time:.4f} seconds.")
#
#     # Stemming step
#     start_time = time.time()
#     stemmed_data = [
#         [ps.stem(word) for word in sentence] for sentence in filtered_data
#     ]
#     end_time = time.time()
#     stemming_time = end_time - start_time
#     total_time += stemming_time
#     print(f"Stemming took {stemming_time:.4f} seconds.")
#
#     print(f"\nTotal preprocessing time: {total_time:.4f} seconds.")
#     return tokenized_data, filtered_data, ner_data, bigram_data, stemmed_data
#
# # Main Function (where you integrate the HTML reading and preprocessing)
# def main():
#     # Directory containing your HTML files (replace with the actual path to your folder)
#     directory = r"C:\Users\grand\PycharmProjects\SearchEngin\videogames"
#
#     # Read HTML files and extract text
#     text_data = read_html_files_from_directory(directory)
#
#     # Run preprocessing and track time for each step
#     tokenized, filtered, ner, bigrams, stemmed = track_time_for_preprocessing(text_data)
#
# # Run the main function
# if __name__ == "__main__":
#     main()


import os
import re
import math
import nltk
import spacy
import pickle
from nltk import ngrams
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import time


# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize SpaCy for NER and WordNet for Lemmatization
nlp = spacy.load('en_core_web_md')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

AGE_RATINGS = ['Everyone', 'Teen', 'Mature', 'TBA', 'Everyone 10+', 'Rating Pending']

#config flags ( for the testing )

CONFIG = {
    'use_stemming': False,
    'use_ner': True,
    'use_bigrams':True,
    'use_metadata':True,
    'use_lemmatization':True,
    'use_stopword_removal':True,
    'use_query_expansion':True
}

WEIGHTS = {
    'title': 3,
    'heading': 2,
    'metadata': 2,
    'ner': 2.5,
    'bigrams':2,
    'body':1
}

def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(int))
    for doc, tokens in documents.items():
        for token in tokens:
            inverted_index[token][doc] += 1
    return inverted_index



# Custom Tokenizer for Hashtags, Mentions, and Game Titles (Multi-Word Entities)
def custom_tokenizer(text):
    pattern = r"@\w+|#\w+|\b[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*\b|[.,!?;]"
    return re.findall(pattern, text)


# Lemmatization function
def lemmatize_tokens(tokens):
    if CONFIG['use_lemmatization']:
        return [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return tokens

def stem_tokens(tokens, text):
    # Check if stemming is enabled
    if CONFIG['use_stemming']:
        # Extract named entities if NER is enabled
        named_entities = []
        if CONFIG['use_ner']:
            named_entities = extract_named_entities(text)  # Extract named entities only if enabled

        # Convert named_entities list to a set for faster lookup
        named_entities_set = set(named_entities)

        # Stem tokens but keep named entities intact
        return [stemmer.stem(token) if token.lower() not in named_entities_set else token for token in tokens]

    return tokens


def extract_named_entities(text):
    if CONFIG['use_ner']:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        print(f"Extracted named entities: {entities}")  # Debugging: Print named entities
        return entities
    return []

def generate_bigrams(tokens,text):
    if CONFIG['use_bigrams']:
        bigrams_list = list(ngrams(tokens,2))
        stemmed_bigrams = list(ngrams(stem_tokens(tokens,text),2))
        return [' '.join(bigram) for bigram in bigrams_list + stemmed_bigrams]
    return []

# Stop word removal
def remove_stopwords(tokens):
    if CONFIG['use_stopword_removal']:
        return [token for token in tokens if token.lower() not in stop_words]
    return tokens
def advanced_tokenizer(text):
    tokens = custom_tokenizer(text)
    print(f"Tokens after custom tokenizer: {tokens}")

    tokens = remove_stopwords(tokens)
    print(f"Tokens after stopword removal: {tokens}")

    tokens = lemmatize_tokens(tokens)
    print(f"Tokens after lemmatization: {tokens}")

    tokens = stem_tokens(tokens, text)  # Apply stemming with named entity preservation
    print(f"Tokens after stemming: {tokens}")

    bigrams = generate_bigrams(tokens,text)
    print(f"Bigrams: {bigrams}")

    return tokens + bigrams

#Function to read and process all HTML files in a directory
def process_html_directory_including_weights(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')

                text = soup.get_text()

                named_entites = extract_named_entities(text)

                title_tokens = advanced_tokenizer(soup.title.get_text()) if soup.title else[]
                heading_tokens = advanced_tokenizer(' '.join([h.get_text() for h in soup.find_all(['h1', 'h2'])]))


                weighted_tokens = title_tokens + heading_tokens + advanced_tokenizer(text)
                weighted_tokens.extend(named_entites * 2)


                documents[filename] = weighted_tokens
    return documents

def process_html_directory(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text()

                #all the named entities
                named_entites = extract_named_entities(text)

                # Extract and tokenize title (weight = 3x)
                title_tokens = advanced_tokenizer(soup.title.get_text()) * int(WEIGHTS['title'])if soup.title else []

                # Extract and tokenize h1, h2 headings (weight = 2x)
                heading_tokens = advanced_tokenizer(' '.join([h.get_text() for h in soup.find_all(['h1', 'h2'])])) * int(WEIGHTS['heading'])


                meta_desc_tokens, meta_keywords_tokens = [], []
                if CONFIG['use_metadata']:
                    # Extract and tokenize meta description (weight = 2x)
                    meta_desc = ' '.join([meta['content'] for meta in soup.find_all('meta', {'name': 'description'})])
                    meta_desc_tokens = advanced_tokenizer(meta_desc) * int(WEIGHTS['metadata'])

                    # Extract and tokenize meta keywords (weight = 2x)
                    meta_keywords = ' '.join([meta['content'] for meta in soup.find_all('meta', {'name': 'keywords'})])
                    meta_keywords_tokens = advanced_tokenizer(meta_keywords) * int(WEIGHTS['metadata'])

                # Tokenize body text (default weight)
                body_tokens = advanced_tokenizer(text)
                bigram_tokens = generate_bigrams(body_tokens) * int(WEIGHTS['bigrams'])

                # Combine all weighted tokens
                weighted_tokens = (title_tokens + heading_tokens +
                                   meta_desc_tokens + meta_keywords_tokens +
                                   body_tokens + bigram_tokens)

                # Double weight for named entities
                weighted_tokens.extend(named_entites * int(WEIGHTS['ner']))

                # Store tokens for each document
                documents[filename] = weighted_tokens

    return documents



#Calculate term frequency (TF)
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


def expand_query(query_tokens):
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        synonyms = set()
        for syn in wn.synsets(token):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        expanded_tokens.update(synonyms)
    return list(expanded_tokens)

# Search for query using VSM
def search_query(query, tf_idf, idf, documents,inverted_index):
    query_tokens = advanced_tokenizer(query) #allows for multi-term query's
    query_entities = extract_named_entities(query)

   # Identify age ratings in the query
    query_age_ratings = [token for token in query_tokens if token in AGE_RATINGS]

    if CONFIG['use_query_expansion']:
        expanded_query_tokens = expand_query(query_tokens)
    else:
        expanded_query_tokens = query_tokens

    query_tf = calculate_tf(expanded_query_tokens)
    query_vector = {token: query_tf[token] * idf.get(token, 0) for token in query_tf}

    candidate_docs = set()
    for token in expanded_query_tokens:
        if token in inverted_index:
            candidate_docs.update(inverted_index[token].keys())

    results = []
    for doc in candidate_docs:
        similarity = cosine_similarity(query_vector, tf_idf.get(doc, {}))

        # Boost if document contains age rating or entities
        if any(age in documents[doc] for age in query_age_ratings):
            similarity += 0.7
        if any(ent in documents[doc] for ent in query_entities):
            similarity += 0.5

        matched_tokens = [token for token in expanded_query_tokens if token in documents[doc]]
        results.append((doc, similarity, matched_tokens))

    results = sorted(results, key=lambda x: x[1], reverse=True)


    # Fill to 10 results by adding low-similarity documents
    if len(results) < 10:
        remaining_docs = set(documents.keys()) - candidate_docs
        for doc in remaining_docs:
            results.append((doc, 0, []))  # Add with zero similarity if not a candidate

    # Sort again to ensure the best results are still at the top
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Always return 10 results (or fewer if fewer documents exist)
    return results


# Save results to a file
def save_results_to_file(results,documents, output_file):
    with open(output_file, 'w') as file:
        # for rank,(doc, score, matches) in enumerate(results, start=1):
        #     file.write(f"Rank: {rank} | {doc}: {score:.4f}, Matches: {', '.join(matches)}\n")
        ranked_docs = {doc: (score, matches) for doc, score, matches in results}
        all_docs = sorted(documents.keys(), key=lambda x: ranked_docs.get(x, (0, []))[0], reverse=True)

        for rank, doc in enumerate(all_docs, start=1):
            score, matches = ranked_docs.get(doc, (0, []))
            file.write(f"Rank: {rank} | {doc}: {score:.4f}, Matches: {', '.join(matches)}\n")


# CLI for the search
def run_search_cli():
    print("Select mode: ")
    print("1. normal Mode")
    print("2. Testing mode (Choose preprocessors)")

    mode = input("Enter 1 or 2: ").strip()


    if mode == '2':
        print('running in Testing mode')

        CONFIG['use_stemming'] = input("Enable stemming? (y/n): ").strip().lower() == 'y'
        CONFIG['use_ner'] = input('Enable NER? (y/n): ').strip().lower() == 'y'
        CONFIG['use_bigrams'] = input("Enable bigrams? (y/n): ").strip().lower() == 'y'
        CONFIG['use_metadata'] = input('Enable metadata? (y/n): ').strip().lower() =='y'
        CONFIG['use_lemmatization'] = input('Enable lemmatization? (y/n): ').strip().lower() == 'y'
        CONFIG['use_stopword_removal'] = input('Enable stopword removal? (y/n): ').strip().lower() == 'y'
        CONFIG['use_query_expansion'] = input('Enable Query Expansion? (y/n): ').strip().lower() == 'y'

    else:
        print('running in Efficiency mode')

    # directory_path = input("Enter the path to the directory containing HTML files: ").strip()
    directory_path = r"C:\Users\grand\PycharmProjects\SearchEngin\videogames"
    print("Processing files...")


    # documents = measure_baseline(directory_path)
    # idf = calculate_idf(documents)
    # tf_idf = calculate_tf_idf(documents,idf)


    pickle_file = 'documents_cache.pkl'

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            loaded_data = pickle.load(file)
            if len(loaded_data) == 3:
                documents,idf,tf_idf = loaded_data
                inverted_index = build_inverted_index(documents)
            else:
                documents,idf,tf_idf,inverted_documents = loaded_data
        print("loaded cache data")
    else:
        documents = process_html_directory(directory_path)
        idf = calculate_idf(documents)
        tf_idf = calculate_tf_idf(documents, idf)
        inverted_index = build_inverted_index(documents)
        with open(pickle_file, 'wb') as file:
            pickle.dump((documents,idf , tf_idf,inverted_index), file)
        print("Files processed successfully. You can now run search queries.")




    while True:
        query = input("\nEnter a search query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        results = search_query(query, tf_idf, idf, documents,inverted_index)
        if results:
            print("\nRanked Results:")
            for rank,  (doc, score, matches) in enumerate(results[:10],start=1):
                file_path = os.path.join(directory_path,doc)
                with open(file_path,'r',encoding='utf-8') as file:
                    soup = BeautifulSoup(file,'html.parser')
                #     content = soup.get_text()
                #     snippet = ""
                #     if matches:
                #         match = re.search(matches[0], content, re.IGNORECASE)
                #         if match:
                #             start = max(0, match.start() - 80)  # 50 characters before match
                #             end = min(len(content), match.end() + 80)  # 50 after match
                #             snippet = content[start:end].replace('\n', ' ')  # Clean up newlines
                #
                # print(f"Rank {rank}: {doc} - Score: {score:.4f}")
                # print(f"  URL: {file_path}")
                # print(f"  Content Snippet: {snippet}...")
                # print(f"  Matches: {', '.join(matches)}")
                # print("-" * 50)  # Separator line for clarity
                content = soup.get_text()

                snippet = ""

                # Extract around the first match for better context
                for match in matches:
                    match_obj = re.search(match, content, re.IGNORECASE)
                    if match_obj:
                        start = max(0, match_obj.start() - 150)  # 150 chars before match
                        end = min(len(content), match_obj.end() + 150)  # 150 after match
                        snippet = content[start:end].replace('\n', ' ')
                        break  # Stop at first match found

                # Fallback to first 500 characters if no match-based snippet is found
                if not snippet:
                    snippet = content[:500].replace('\n', ' ').strip()

                print(f"Rank {rank}: {doc} - Score: {score:.4f}")
                print(f"  URL: {file_path}")
                print(f"  Content Snippet: {snippet[:300]}...")  # Limit to 300 characters
                print(f"  Matches: {', '.join(matches)}")
                print("-" * 50)
            save_results_to_file(results,documents, "search_results.txt")
            print("\nResults saved to 'search_results.txt'.")
        else:
            print("No matches found.")

if __name__ == "__main__":
    run_search_cli()