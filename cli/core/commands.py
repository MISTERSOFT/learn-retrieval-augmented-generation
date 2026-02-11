from core.text_processing import text_processing_pipeline, has_matching_token
from core.utils import load_file
from nltk.stem import PorterStemmer


def search_command(query: str, limit: int = 5):
    movies_json = load_file("data/movies.json", "json")

    stopwords = load_file("data/stopwords.txt", "txt")
    stopwords = stopwords.splitlines()

    stemmer = PorterStemmer()

    processed_query = text_processing_pipeline(query, stopwords, stemmer)

    results = []

    for movie in movies_json["movies"]:
        processed_movie_title = text_processing_pipeline(
            movie["title"], stopwords, stemmer
        )

        if has_matching_token(processed_query, processed_movie_title):
            results.append(movie)
            if len(results) >= limit:
                break

    return results
