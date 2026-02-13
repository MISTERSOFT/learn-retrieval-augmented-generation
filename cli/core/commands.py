import math
from core.inverted_index import InvertedIndex
from core.text_processing import text_processing_pipeline, has_matching_token
from core.utils import load_file
from nltk.stem import PorterStemmer

movies_json = load_file("data/movies.json", "json")

stopwords = load_file("data/stopwords.txt", "txt")
stopwords = stopwords.splitlines()

stemmer = PorterStemmer()

def search_command(query: str, limit: int = 5):
    inverted_index = InvertedIndex(stopwords, stemmer)
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    processed_query = text_processing_pipeline(query, stopwords, stemmer)

    results = []

    for q_token in processed_query:
        docs = inverted_index.get_documents(q_token)
        for doc in docs:
            results.append(doc)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return results

def build_command():
    # Build the Inverted Index
    inverted_index = InvertedIndex(stopwords, stemmer)
    inverted_index.build(movies_json['movies'])
    inverted_index.save()

def tf_command(doc_id: int, term: str):
    inverted_index = InvertedIndex(stopwords, stemmer)
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    return inverted_index.get_tf(doc_id, term)

def idf_command(term):
    inverted_index = InvertedIndex(stopwords, stemmer)
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    tokenized_term = text_processing_pipeline(term, stopwords, stemmer)
    if len(tokenized_term) > 1:
        raise Exception("Too many terms has been given. Expected one.")

    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(tokenized_term[0]))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def tfidf_command(doc_id: int, term: str):
    inverted_index = InvertedIndex(stopwords, stemmer)
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    term_frequency = inverted_index.get_tf(doc_id, term)

    tokenized_term = text_processing_pipeline(term, stopwords, stemmer)
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(tokenized_term[0]))
    inverse_document_frequency = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return term_frequency * inverse_document_frequency