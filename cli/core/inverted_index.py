from collections import Counter
from pathlib import Path
import pickle
from typing import Any
from nltk.stem import PorterStemmer

from core.text_processing import text_processing_pipeline


class InvertedIndex:
    index: dict[str, list[int]]
    docmap: dict[int, Any]
    term_frequencies: dict[int, Counter]
    __stopwords: list[str]
    __stemmer: PorterStemmer

    def __init__(self, stopwords, stemmer):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.__stopwords = stopwords
        self.__stemmer = stemmer

    def __add_document(self, doc_id: int, text: str):
        tokens = text_processing_pipeline(text, self.__stopwords, self.__stemmer)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in tokens:
            if self.index.get(token) is None:
                self.index[token] = [doc_id]
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)

    def get_documents(self, term: str):
        doc_ids = self.index.get(term)
        if doc_ids is None or not doc_ids:
            return []
        sorted_doc_ids = sorted(doc_ids)
        docs = []
        for doc_id in sorted_doc_ids:
            docs.append(self.docmap[doc_id])
        return docs

    def get_tf(self, doc_id, term):
        tokenized_term = text_processing_pipeline(term, self.__stopwords, self.__stemmer)
        if len(tokenized_term) > 1:
            raise Exception("Too many terms has been given. Expected one.")
        return self.term_frequencies[doc_id][tokenized_term[0]]

    def build(self, movies):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
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

    def load(self):
        with Path('cache/index.pkl').open('rb') as file:
            self.index = pickle.load(file)

        with Path('cache/docmap.pkl').open('rb') as file:
            self.docmap = pickle.load(file)

        with Path('cache/term_frequencies.pkl').open('rb') as file:
            self.term_frequencies = pickle.load(file)