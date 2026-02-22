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

BM25_K1 = 1.5
BM25_B = 0.75


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
    # Split text into token array, also remove whitespaces (space, tab, etc.)
    tokens = text.split()
    # Remove empty strings
    tokens = [text for text in tokens if text]
    return tokens

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
    doc_lengths: dict[int, int]

    __cache_path: Path
    __index_cache_file_path: Path
    __docmap_cache_file_path: Path
    __term_frequencies_cache_file_path: Path
    __doc_lengths_cache_file_path: Path

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()

        self.__cache_path = Path('cache')
        self.__index_cache_file_path = self.__cache_path.joinpath('index.pkl')
        self.__docmap_cache_file_path = self.__cache_path.joinpath('docmap.pkl')
        self.__term_frequencies_cache_file_path = self.__cache_path.joinpath('term_frequencies.pkl')
        self.__doc_lengths_cache_file_path = self.__cache_path.joinpath('doc_lengths.pkl')

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        # Transform tokens list to a set, it prevent token duplication
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

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

        Formula: log(N / df)

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

        total_doc_count = len(self.docmap) # N
        term_match_doc_count = len(self.index[tokenized_term[0]]) # df

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

    def get_bm25_idf(self, term: str) -> float:
        """Calculate the Okapi BM25 IDF score.

        Formula: log( (N - df + 0.5) / (df + 0.5) + 1 )

        Args:
            term (str): Term to search

        Raises:
            Exception: Too many terms has been given, only one required

        Returns:
            float: IDF score
        """
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Too many terms has been given. Expected one.")

        N = len(self.docmap) # documents count
        df = len(self.index[tokenized_term[0]]) # document frequency (how many docs contains the term)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        """Calculate the BM25 Term Frequency.

        Args:
            doc_id (int): Document ID
            term (str): Term to search
            k1 (float, optional): Control the diminishing return. Defaults to BM25_K1.

        Returns:
            float: BM25 Term Frequency score
        """
        tf = self.get_tf(doc_id, term)

        # Length normalization factor
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str):
        """Return a BM25 score.

        Args:
            doc_id (int): Document ID
            term (str): Term to search

        Returns:
            float: BM25 score
        """
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int):
        """Use BM25 to search relevant results.

        Args:
            query (str): User search query
            limit (int): Number of maximum result

        Returns:
            list: Matched movies
        """
        qtokens = tokenize_text(query)
        scores = dict()

        for doc_id in self.docmap:
            score = 0.0
            for qtoken in qtokens:
                score += self.bm25(doc_id, qtoken)
            scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, doc_score in sorted_scores[:limit]:
            results.append((self.docmap[doc_id], doc_score))
        return results

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
        # cache_path = Path('cache')
        try:
            # cache_path.mkdir()
            self.__cache_path.mkdir()
        except FileExistsError:
            print(f"Directory '{self.__cache_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{self.__cache_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Dump index and docmap into the cache folder, with highest protocol for best performance
        # with Path('cache/index.pkl').open('wb') as file:
        with self.__index_cache_file_path.open('wb') as file:
            pickle.dump(self.index, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{self.__index_cache_file_path} dumped !')

        # with Path('cache/docmap.pkl').open('wb') as file:
        with self.__docmap_cache_file_path.open('wb') as file:
            pickle.dump(self.docmap, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{self.__docmap_cache_file_path} dumped !')

        # with Path('cache/term_frequencies.pkl').open('wb') as file:
        with self.__term_frequencies_cache_file_path.open('wb') as file:
            pickle.dump(self.term_frequencies, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{self.__term_frequencies_cache_file_path} dumped !')

        with self.__doc_lengths_cache_file_path.open('wb') as file:
            pickle.dump(self.doc_lengths, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{self.__doc_lengths_cache_file_path} dumped !')

    def load(self) -> None:
        """Load cache."""

        # with Path('cache/index.pkl').open('rb') as file:
        with self.__index_cache_file_path.open('rb') as file:
            self.index = pickle.load(file)
        print(f'{self.__index_cache_file_path} loaded !')

        # with Path('cache/docmap.pkl').open('rb') as file:
        with self.__docmap_cache_file_path.open('rb') as file:
            self.docmap = pickle.load(file)
        print(f'{self.__docmap_cache_file_path} loaded !')

        # with Path('cache/term_frequencies.pkl').open('rb') as file:
        with self.__term_frequencies_cache_file_path.open('rb') as file:
            self.term_frequencies = pickle.load(file)
        print(f'{self.__term_frequencies_cache_file_path} loaded !')

        with self.__doc_lengths_cache_file_path.open('rb') as file:
            self.doc_lengths = pickle.load(file)
        print(f'{self.__doc_lengths_cache_file_path} loaded !')