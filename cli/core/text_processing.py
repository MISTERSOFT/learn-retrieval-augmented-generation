import math
import string
from core.utils import load_file
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
from pathlib import Path
import pickle
from typing import Any


PUNCTUATION_TRANS_TABLE = str.maketrans("", "", string.punctuation)

stopwords = load_file("data/stopwords.txt", "txt").splitlines()
stemmer = PorterStemmer()

def make_case_insensitive(text: str) -> list[str]:
    """
    Step 1 out of 5 of the tokenization pipeline.

    Convert all text to lowercase.

    Args:
        text (str): String containing words

    Returns:
        str: Text lowered
    """
    return text.lower()

def remove_punctuations(text: str) -> list[str]:
    """
    Step 2 out of 5 of the tokenization pipeline.

    Remove punctuation (periods, commas, etc...).

    Example: "hello, world!" becomes "hello world"

    Args:
        text (str): String containing words

    Returns:
        str: Text without punctuations
    """
    return text.translate(PUNCTUATION_TRANS_TABLE)

def tokenize(text: str) -> list[str]:
    """
    Step 3 out of 5 of the tokenization pipeline.

    Break text into individual words.

    Example: "hello world" becomes ["hello", "world"]

    Args:
        text (str): String containing words

    Returns:
        list[str]: List of tokens
    """
    # Replace whitespaces into single space
    token = text.replace(r"\s+", " ")
    # Remove whitespaces from edges
    token = text.strip()
    # Split text into token array
    token = text.split(" ")
    return token

def remove_stop_words(tokens: list[str], stopwords: list[str]) -> list[str]:
    """
    Step 4 out of 5 of the tokenization pipeline.

    Remove common stop words that don't add much meaning (a, an, of, etc...).

    Example: ["a", "puppy"] becomes ["puppy"]

    Args:
        tokens (list[str]): List of tokens
        stopwords (list[str]): List of stop words

    Returns:
        list[str]: List of tokens without stop words
    """
    return [token for token in tokens if token not in stopwords]

def stem(tokens: list[str]) -> list[str]:
    """
    Step 5 out of 5 of the tokenization pipeline.

    Keep only the stem of words.

    Example: ["running", "jumping"] becomes ["run", "jump"]

    Args:
        tokens (list[str]): List of words

    Returns:
        list[str]: List of words stemmed
    """
    return [stemmer.stem(token) for token in tokens]

def tokenize_text(query: str) -> list[str]:
    """
    Transform a text to tokens.

    Args:
        query (str): List of words

    Returns:
        list[str]: Tokens
    """
    q = make_case_insensitive(query)
    q = remove_punctuations(q)
    tokens = tokenize(q)
    tokens = remove_stop_words(tokens, stopwords)
    tokens = stem(tokens)
    return tokens

def has_matching_token(query_tokens: list[str], movie_title_tokens: list[str]) -> bool:
    """
    Search for matching at least one token from user query tokens to the movie title tokens.

    Args:
        query_tokens (list[str]): User query tokens
        movie_title_tokens (list[str]): Movie title tokens

    Returns:
        bool: True if a token matched
    """
    for mt_token in movie_title_tokens:
        for q_token in query_tokens:
            if q_token in mt_token:
                return True
    return False



class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Any]
    term_frequencies: dict[int, Counter]

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = dict()
        self.term_frequencies = dict()

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        # Transform tokens list to a set, it prevent token duplication
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term: str):
        """Get document from a term.

        Args:
            term (str): Term to search

        Returns:
            list: List of document with the given term
        """
        doc_ids = self.index.get(term, set())
        return [self.docmap[doc_id] for doc_id in sorted(doc_ids)]

    def get_tf(self, doc_id, term) -> int:
        """Get the Term Frequency of a document.

        Args:
            doc_id (int): Document ID
            term (str): Term to determine frequency

        Raises:
            Exception: Too many terms. Only one term is required.

        Returns:
            int: Frenquency of the term
        """
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Too many terms has been given. Expected one.")
        return self.term_frequencies[doc_id][tokenized_term[0]]

    def get_idf(self, term: str) -> float:
        """Compute IDF (Inverse Document Frequency) of a term.

        Args:
            term (str): Term use for determine the IDF

        Raises:
            Exception: Too many terms, only one expected.

        Returns:
            float: IDF score
        """
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Too many terms has been given. Expected one.")

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[tokenized_term[0]])

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        """Get the TF-IDF (Term Frequency-Inverse Document Frequency) of document.

        Args:
            doc_id (int): Document ID
            term (str): Term use for TF-IDF calculation

        Returns:
            float: Score
        """
        term_frequency = self.get_tf(doc_id, term)
        inverse_document_frequency = self.get_idf(term)
        return term_frequency * inverse_document_frequency

    def build(self, movies) -> None:
        """Build Inverted Indexes.

        Args:
            movies: List of movies
        """
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self) -> None:
        """Cache the Inverted Indexes"""

        # Create cache directory if not exists
        cache_path = Path('cache')
        try:
            cache_path.mkdir()
        except FileExistsError:
            print(f"Directory '{cache_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{cache_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Dump index and docmap into the cache folder, with highest protocol for best performance
        with Path('cache/index.pkl').open('wb') as file:
            pickle.dump(self.index, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('cache/index.pkl dumped !')

        with Path('cache/docmap.pkl').open('wb') as file:
            pickle.dump(self.docmap, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('cache/docmap.pkl dumped !')

        with Path('cache/term_frequencies.pkl').open('wb') as file:
            pickle.dump(self.term_frequencies, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('cache/term_frequencies.pkl dumped !')

    def load(self) -> None:
        """Load cache."""

        with Path('cache/index.pkl').open('rb') as file:
            self.index = pickle.load(file)
        print('cache/index.pkl loaded !')

        with Path('cache/docmap.pkl').open('rb') as file:
            self.docmap = pickle.load(file)
        print('cache/docmap.pkl loaded !')

        with Path('cache/term_frequencies.pkl').open('rb') as file:
            self.term_frequencies = pickle.load(file)
        print('cache/term_frequencies.pkl loaded !')