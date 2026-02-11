import string
from nltk.stem import PorterStemmer


PUNCTUATION_TRANS_TABLE = str.maketrans("", "", string.punctuation)

def make_case_insensitive(query: str):
    '''
    Step 1 out of 5 of the text processing pipeline.

    Convert all text to lowercase.

    :param query: String containing words
    :type query: str
    '''
    return query.lower()

def remove_punctuations(query: str):
    '''
    Step 2 out of 5 of the text processing pipeline.

    Remove punctuation (periods, commas, etc...).

    Example: "hello, world!" becomes "hello world"

    :param query: String containing words
    :type query: str
    '''
    return query.translate(PUNCTUATION_TRANS_TABLE)

def tokenize(query: str):
    '''
    Step 3 out of 5 of the text processing pipeline.

    Break text into individual words.

    Example: "hello world" becomes ["hello", "world"]

    :param query: String containing words
    :type query: str
    '''
    # Replace whitespaces into single space
    token = query.replace(r"\s+", " ")
    # Remove whitespaces from edges
    token = query.strip()
    # Split query into token array
    token = query.split(" ")
    return token

def remove_stop_words(tokens: list[str], stopwords: list[str]):
    '''
    Step 4 out of 5 of the text processing pipeline.

    Remove common stop words that don't add much meaning (a, an, of, etc...).

    Example: ["a", "puppy"] becomes ["puppy"]

    :param tokens: List of words
    :type tokens: list[str]
    :param stopwords: List of stop words
    :type stopwords: list[str]
    '''
    return [token for token in tokens if token not in stopwords]

def stem(tokens: list[str], stemmer: PorterStemmer):
    '''
    Step 5 out of 5 of the text processing pipeline.

    Keep only the stem of words.

    Example: ["running", "jumping"] becomes ["run", "jump"]

    :param tokens: List of words
    :type tokens: list[str]
    :param stemmer: Stemmer instance
    :type stemmer: PorterStemmer
    '''
    return [stemmer.stem(token) for token in tokens]

def text_processing_pipeline(query: str, stopwords: list[str], stemmer: PorterStemmer) -> list[str]:
    '''
    Process a text to tokens.

    :param query: List of words
    :type query: str
    :param stopwords: List of stop words
    :type stopwords: list[str]
    :param stemmer: Stemmer instance
    :type stemmer: PorterStemmer
    '''
    q = make_case_insensitive(query)
    q = remove_punctuations(q)
    tokens = tokenize(q)
    tokens = remove_stop_words(tokens, stopwords)
    tokens = stem(tokens, stemmer)
    return tokens

def has_matching_token(query_tokens: list[str], movie_title_tokens: list[str]) -> bool:
    '''
    Search for matching at least one token from user query tokens to the movie title tokens.

    :param query_tokens: User query tokens
    :type query_tokens: list[str]
    :param movie_title_tokens: Movie title tokens
    :type movie_title_tokens: list[str]
    :return: True if a token matched
    :rtype: bool
    '''
    for mt_token in movie_title_tokens:
        for q_token in query_tokens:
            if q_token in mt_token:
                return True
    return False