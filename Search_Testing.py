
# def evaluate_search(query, retrieved_results, relevant_documents):
#     relevant_retrieved = [doc for doc, _, _ in retrieved_results if doc in relevant_documents]
#
#     precision = len(relevant_retrieved) / len(retrieved_results) if len(retrieved_results) > 0 else 0
#     recall = len(relevant_retrieved) / len(relevant_documents) if len(relevant_documents) > 0 else 0
#
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
#
#     return precision, recall, f1_score

# relevant_docs = ['game1.html', 'game2.html']  # Define relevant documents manually
# results = search_query("Sports Genre Games", tf_idf, idf, documents)
# precision, recall, f1 = evaluate_search("Sports Genre Games", results, relevant_docs)
# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Set up NER using spacy
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

# Sample Configuration
CONFIG = {
    'use_stopwords': True,
    'use_stemming': True,
    'use_lemmatization': True,
    'use_ner': True,  # Make sure this is True if you want NER
}


# Custom tokenizer and preprocessing functions

def custom_tokenizer(text):
    # Tokenizing using word_tokenize from nltk
    tokens = word_tokenize(text)

    # Stopword removal
    if CONFIG['use_stopwords']:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization
    if CONFIG['use_lemmatization']:
        tokens = [token.lower() for token in tokens]  # Simple lemmatization as an example

    # Stemming
    if CONFIG['use_stemming']:
        tokens = stem_tokens(tokens, text)

    # Print tokens after all processing
    print("Tokens after stemming:", tokens)

    # Named Entity Recognition (NER)
    if CONFIG['use_ner']:
        named_entities = extract_named_entities(text)
        print("Named Entities:", named_entities)

    return tokens


def stem_tokens(tokens, text):
    # Check if stemming is enabled
    named_entities = []
    if CONFIG['use_ner']:
        named_entities = extract_named_entities(text)  # Extract named entities if enabled

    named_entities_set = set(named_entities)  # Create a set for faster lookup

    # Apply stemming but keep named entities intact
    return [stemmer.stem(token) if token.lower() not in named_entities_set else token for token in tokens]


def extract_named_entities(text):
    # Extract named entities only if the flag is set to True
    if CONFIG['use_ner']:
        doc = nlp(text)
        return [ent.text for ent in doc.ents]  # Return entities as a list
    return []


# Example Query
query = "games published by Activision"

# Run the custom tokenizer
tokens = custom_tokenizer(query)

print(tokens)
