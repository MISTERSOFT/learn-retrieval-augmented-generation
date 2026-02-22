from typing import Optional
from core.text_processing import tokenize_text, InvertedIndex
from core.utils import load_file


def search_command(query: str, limit: int = 5):
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    processed_query = tokenize_text(query)

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
    """Build the Inverted Index."""

    movies_json = load_file("data/movies.json", "json")
    inverted_index = InvertedIndex()
    inverted_index.build(movies_json['movies'])
    inverted_index.save()

def tf_command(doc_id: int, term: str):
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    return inverted_index.get_tf(doc_id, term)

def idf_command(term: str):
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    return inverted_index.get_idf(term)

def tfidf_command(doc_id: int, term: str):
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    return inverted_index.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    return inverted_index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: Optional[float], b: Optional[float]) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)

    return inverted_index.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = 5):
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except:
        print('Unable to load indexes. Exiting program...')
        exit(1)
    return inverted_index.bm25_search(query, limit)