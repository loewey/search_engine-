import math
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import regex as re
from nltk.stem import WordNetLemmatizer





nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def read_from_file(directory_path):
    results = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            filename = os.path.join(directory_path, filename)
            with open(filename,'r',encoding='utf-8') as f:
                soup = BeautifulSoup(f,'html.parser')
                text = soup.get_text()
                tokens = further_tokenizer(text)
                results[filename] = tokens
    return results


def simple_tokenizer(text):
    custom_pattern = r"@\w+|\b[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*\b|[.,!?;]"
    return re.findall(custom_pattern,text)

def apply_ner(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def lemmatize_tokens(tokens):
    return (lemmatizer.lemmatize(token) for token in tokens)

def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stop_words]

def further_tokenizer(text):
    soup = BeautifulSoup(text, 'html.parser')
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        cleanedParagraphs = p.get_text()
        tokens = simple_tokenizer(cleanedParagraphs)

        entities = apply_ner(text)

        for entity in entities:
            tokens = [token if token.lower() != entity.lower() else f"<ENTITY : {entity}" for token in tokens]

        tokens = remove_stopwords(tokens)

        tokens = lemmatize_tokens(tokens)

    return tokens


