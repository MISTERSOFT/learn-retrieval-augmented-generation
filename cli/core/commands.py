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