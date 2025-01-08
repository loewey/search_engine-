# Table of contents 

1. Prerequisites 
2. Installation
3. Configuration
4. Running the project 
5. Testing
6. Usage 
7. troubleshooting and FAQ 
8. Contributions 

# summary 
The Search engine was created with the primary goal to search and return 10  
ranked results based on user query inputs, with the initial dataset of "Videogames" in mind whilst analysing each 
preprocessor's efficiency and overall use 


---
# Prerequisites 

Ensure before running the code you have the following: 

- Python 3.8 or higher 
- pip (Python package installer)
### Required Python libraries 

- nltk 
- spacy 
- bs4 (BeautifulSoup4)
- pickle
additionally, download the necessary NLTK resources and SpaCy model 
> `python -m spacy download en_core_web_md`
---
# installation 

1. If you are on Github, Clone the repository :
>`git clone https://github.com/loowy/search_engin.git`
2. Download required NLTK resources 
 ```
> import nltk 
> nltk.download('punkt')
> nltk.download('stopwords')
> nltk.download('wordnet') 
```
---
# Configuration

To alter the behavior and results in "normal" mode, you can modify the 
CONFIG dictionary in the code. Options include : 
```
CONFIG = {
    'use_stemming': True,
    'user_ner': True,
    'use_bigrams': True,
    'use_metadata': True,
    'use_lemmatization': True,
    'use_stopword_removal': True,
    'use_query_expansion': True
}
```
By modifying each flag as needed to customize search engine behaviour.

--- 
# Running the project 

To run the search engine project, execute the following command in termal: 
> `python Search_engine.py`

alternatively you can open the file in an IDE of your choosing and run the 
file from there. 

1. Choose the mode : 
- 1. Normal mode (Default configuration - fastest preprocessor combination)
- 2. Testing mode (Custom configuration set by user)
3. Provide the path to the directory containing HTML files. 
4. Enter a search query to retrieval ranked results. 

---

# testing 
to test the engine with different configurations (preprocesors)
>`python Search_engine.py`

Select testing Mode and enable/disable features interactively 
- Stemming 
- Named Entity Recognition (NER)
- Bigrams 
- Metadata extraction 
- Lemmatization 
- Stopword removal 
- Query Expansion 
---
# usage 
Example : 
```
> Enter the path to the directory containing HTML files : /path/to/html/files
> Enter a search query : Hollow Knight 
```
Results will display the top 10 ranked documents with snippets and be saved to 
<mark>search_results.txt</mark>

--- 
# Troubleshooting and FAQ 

### 1. Error: <mark> OSError:[E050] Can't find model 'en_core_web_mb'
***solution :*** Run the following command to download the required SpaCy model: 
> `python -m spacy download en_core_web_mb`
### 2. Error <mark"> ModuleNotFoundError: No module named 'bs4'
***solution :*** Ensure that BeautifulSoup is installed by running the command : 
> `pip install beautifulsoup4`
### 3. HTML files not processing/producing empty results : 
***solution :*** Check that the directory provided contains <mark>.html</mark>
files. Ensure the files have proper <mark>&lt;title&gt;</mark>, <mark>&lt;h1&gt;</mark> 
or <mark>&lt;body&gt;</mark> tags for processing. 

### 4. Cached results are outdated 
***solution :*** Delete the <mark>documents_cache.pkl</mark> file in the project directory. 
The cache will rebuild the next time that the program is run 

### 5. No Search engine results returned
***solution :*** Ensure that the HTML files contain relevant content and that the search query 
includes terms that are likely to be found in the documents in the dataset 

---
# Contribution 

Contributions are welcome. If you wish to follow the steps : 
1. fork the project 
2. Create a feature branch : 
> `git checkout -b feature/new-feature`
3. commit changes 
> `git commit -m "new changed"`
4. push to branch and open a PR 
--- 


<style>
mark{
    background-color:grey;
    color: seashell;
}
</style>


