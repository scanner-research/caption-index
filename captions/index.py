"""
Indexes for srt files
"""

import csv
import pickle
from abc import ABC, abstractmethod, abstractproperty
from collections import deque
from typing import (
    Dict, Iterable, List, Set, Tuple, NamedTuple, Union, Optional, Generator)

from .lemmatize import default_lemmatizer
from .tokenize import default_tokenizer, Tokenizer
from .rs_captions import RsCaptionIndex                 # type: ignore


class Lexicon(object):
    """
    A map from word to id, and vice versa

    Lookup a token with bracket notation:
        w = lexicon["hello"]
        w = lexicon[100]        # Word with id 100

    Iterate the lexicon:
        for w in lexicon:
            ...

    Test if a word is in the lexicon:
        if 'hello' in lexicon:
            ...
    """

    UNKNOWN_TOKEN = '<UNKNOWN>'

    class Word(NamedTuple):
        id: int         # Token id
        token: str      # String representation
        count: int      # Number of occurrences

    class WordDoesNotExist(Exception):
        pass

    WordIdOrString = Union[str, int]

    def __init__(self, words):
        """List of words, where w.id is the index in the list"""
        assert isinstance(words, list)
        self._words = words
        self._inverse = {}
        for i, w in enumerate(words):
            assert w.id == i
            self._inverse[w.token] = w
        self._lemmas = None
        self._lemmatizer = None

    def __iter__(self):
        # Iterate lexicon in id order
        return self._words.__iter__()

    def __getitem__(self, key: WordIdOrString) -> 'Lexicon.Word':
        if isinstance(key, int):
            # Get word by id
            try:
                return self._words[key]
            except IndexError:
                raise Lexicon.WordDoesNotExist('id={}'.format(key))
        elif isinstance(key, str):
            # Get word by token
            try:
                return self._inverse[key]
            except KeyError:
                raise Lexicon.WordDoesNotExist(key)
        raise TypeError('Not supported for {}'.format(type(key)))

    def __contains__(self, key: WordIdOrString) -> bool:
        try:
            self.__getitem__(key)
        except Lexicon.WordDoesNotExist:
            return False
        return True

    def __len__(self) -> int:
        return len(self._words)

    def __lemmas_init(self) -> None:
        """Compute lemmas for every word"""
        lemmatizer = default_lemmatizer()
        lemmas = {}
        for w in self._words:
            for lem in lemmatizer.lemma(w.token.lower()):
                if lem not in lemmas:
                    lemmas[lem] = set()
                lemmas[lem].add(w.id)
        self._lemmatizer = lemmatizer
        self._lemmas = lemmas

    def similar(self, key: WordIdOrString) -> Set[int]:
        """Return words that are similar (share the same lemma)"""
        if self._lemmatizer is None:
            self.__lemmas_init()
            assert self._lemmatizer is not None
            assert self._lemmas is not None
        if isinstance(key, str):
            s = key
        elif isinstance(key, Lexicon.Word):
            s = key.token
        else:
            s = self.__getitem__(key).token
        results = set()
        for lem in self._lemmatizer.lemma(s.lower()):
            results.update(self._lemmas.get(lem, []))
        return results

    def decode(self, key: WordIdOrString,
               default: Optional[str] = None) -> str:
        try:
            return self.__getitem__(key).token
        except Lexicon.WordDoesNotExist:
            return default if default is not None else Lexicon.UNKNOWN_TOKEN

    def store(self, path: str) -> None:
        """Save the lexicon as a TSV file"""
        prev_w = None
        for w in self._words:
            if prev_w:
                assert w.id > prev_w.id, 'Bad lexicon, not sorted by id'
                assert w.token > prev_w.token, \
                    'Bad lexicon, not sorted by token'
            prev_w = w

        with open(path, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for w in self._words:
                tsv_writer.writerow([w.id, w.count, w.token])

    @staticmethod
    def load(path: str) -> 'Lexicon':
        """Load a TSV formatted lexicon"""
        with open(path, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            words = []
            for row in tsv_reader:
                id_, count, token = row
                words.append(Lexicon.Word(id=int(id_), count=int(count),
                                          token=token))
        return Lexicon(words)


class Documents(object):
    """
    A mapping from document id to name, and vice versa

    Lookup documents by name or id with bracket notation:
        d = documents["filename"]
        d = documents[100]

    Iterate the documents:
        for d in documents:
            ...

    Test if a document is in the list:
        if 'test.srt' in documents:
            ...
    """

    class Document(NamedTuple):
        id: int
        name: str

    class DocumentDoesNotExist(Exception):
        pass

    DocumentIdOrName = Union[int, str]

    def __init__(self, docs: List['Documents.Document']):
        """List of Documents, where index is the id"""
        assert all(i == d.id for i, d in enumerate(docs))
        self._docs = docs

    def __iter__(self) -> Iterable['Documents.Document']:
        return self._docs.__iter__()

    def __getitem__(
        self, key: 'Documents.DocumentIdOrName'
    ) -> 'Documents.Document':
        if isinstance(key, int):
            # Get doc name by id (IndexError)
            try:
                return self._docs[key]
            except IndexError:
                raise Documents.DocumentDoesNotExist('id={}'.format(key))
        elif isinstance(key, str):
            # Get doc id by name (KeyError)
            for d in self._docs:
                if d.name == key:
                    return d
            else:
                raise Documents.DocumentDoesNotExist(key)
        raise TypeError('Not supported for {}'.format(type(key)))

    def __contains__(self, key: 'Documents.DocumentIdOrName') -> bool:
        try:
            self.__getitem__(key)
        except Documents.DocumentDoesNotExist:
            return False
        return True

    def __len__(self) -> int:
        return len(self._docs)

    def prefix(self, key: str) -> List['Documents.Document']:
        results = []
        for d in self._docs:
            if d.name.startswith(key):
                results.append(d)
        return results

    def store(self, path: str) -> None:
        """Save the document list as TSV formatted file"""
        with open(path, 'w') as f:
            for d in self._docs:
                f.write('\t'.join([str(d.id), d.name]))
                f.write('\n')

    @staticmethod
    def load(path: str) -> 'Documents':
        """Load a TSV formatted list of documents"""
        documents = []
        with open(path, 'r') as f:
            for line in f:
                i, name = line.strip().split('\t', 1)
                documents.append(Documents.Document(id=int(i), name=name))
        return Documents(documents)


class CaptionIndex(object):
    """
    Interface to a binary encoded index file.
    """

    # A "posting" is an occurance of a token or n-gram
    class Posting(NamedTuple):
        start: float        # Start time in seconds
        end: float          # End time in seconds
        idx: int            # Start position in document
        len: int            # Number of tokens

    # Document object with postings
    class Document(NamedTuple):
        id: int                                 # Document id
        postings: List['CaptionIndex.Posting']  # List of locations

    DocIdOrDocument = Union[int, Documents.Document]
    WordIdOrWord = Union[int, Lexicon.Word]

    def __init__(self, path: str, lexicon: Lexicon, documents: Documents,
                 binary_format: Optional['BinaryFormat'] = None,
                 tokenizer: Optional[Tokenizer] = None, debug: bool = False):
        assert isinstance(lexicon, Lexicon)
        assert isinstance(documents, Documents)
        self._lexicon = lexicon
        self._documents = documents
        self._tokenizer = tokenizer

        if binary_format is None:
            binary_format = BinaryFormat.default()

        self._rs_index = RsCaptionIndex(
            path, datum_size=binary_format.datum_bytes,
            start_time_size=binary_format._start_time_bytes,
            end_time_size=binary_format._end_time_bytes,
            debug=debug)

    def __require_open_index(f):
        def wrapper(self, *args, **kwargs):
            if self._rs_index is None:
                raise ValueError('I/O on closed CaptionIndex')
            return f(self, *args, **kwargs)
        return wrapper

    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = default_tokenizer()
        return self._tokenizer

    @__require_open_index
    def document_length(self, doc: 'CaptionIndex.DocIdOrDocument') -> int:
        """Get the length of a document in tokens"""
        doc_id = self.__get_document_id(doc)
        return self._rs_index.document_length(doc_id)[0]

    @__require_open_index
    def document_duration(self, doc: 'CaptionIndex.DocIdOrDocument') -> float:
        """Get the duration of a document in seconds"""
        doc_id = self.__get_document_id(doc)
        return self._rs_index.document_length(doc_id)[1]

    def search(
        self, text: Union[str, List[WordIdOrWord]],
        documents: Optional[Iterable['CaptionIndex.DocIdOrDocument']] = None
    ) -> Iterable['CaptionIndex.Document']:
        """
        Search for instances of text

        Usage:
            text: string, list of words, or list of word ids
            documents: list of documents or ids to search in
                       ([] or None means all documents)
        """
        if isinstance(text, str):
            tokens = self.__tokenize_text(text)
        else:
            tokens = text
        return self.ngram_search(*tokens, documents=documents)

    @__require_open_index
    def ngram_search(
        self, first_word: Union[
            'CaptionIndex.WordIdOrWord',
            List['CaptionIndex.WordIdOrWord']
        ],
        *other_words,
        documents: Optional[Iterable['CaptionIndex.DocIdOrDocument']] = None
    ) -> Iterable['CaptionIndex.Document']:
        """Search for ngram instances"""
        doc_ids = self.__get_document_ids(documents)
        if len(other_words) == 0:
            result = self._rs_index.unigram_search(
                [w.id for w in self.__get_word_opts(first_word)], doc_ids)
        else:
            ngram_word_ids, anchor_idx = self.__get_ngram_ids_and_anchor_idx(
                [first_word, *other_words])
            result = self._rs_index.ngram_search(
                ngram_word_ids, doc_ids, anchor_idx)
        return self.__unpack_rs_search(result)

    def contains(
        self, text: Union[str, List['CaptionIndex.WordIdOrWord']],
        documents: Optional[Iterable['CaptionIndex.DocIdOrDocument']] = None
    ) -> Set[int]:
        """
        Find documents (ids) containing the text

        Usage:
            text: string, list of words, or list of word ids
            documents: list of documents or ids to search in
                       ([] or None means all documents)
        """
        if isinstance(text, str):
            tokens = self.__tokenize_text(text)
        else:
            tokens = text
        return self.ngram_contains(*tokens, documents=documents)

    @__require_open_index
    def ngram_contains(
        self, first_word: Union[
            'CaptionIndex.WordIdOrWord',
            List['CaptionIndex.WordIdOrWord']
        ],
        *other_words,
        documents: Optional[Iterable['CaptionIndex.DocIdOrDocument']] = None
    ) -> Set[int]:
        """Find documents (ids) containing the ngram"""
        doc_ids = self.__get_document_ids(documents)
        if len(other_words) == 0:
            result = self._rs_index.unigram_contains(
                [w.id for w in self.__get_word_opts(first_word)], doc_ids)
        else:
            ngram_word_ids, anchor_idx = self.__get_ngram_ids_and_anchor_idx(
                [first_word, *other_words])
            result = self._rs_index.ngram_contains(
                ngram_word_ids, doc_ids, anchor_idx)
        return set(result)

    @__require_open_index
    def tokens(
        self, doc: 'CaptionIndex.DocIdOrDocument',
        index: int = 0, count: int = 2 ** 31
    ) -> List[int]:
        """Get token ids for a range of positions in a document"""
        doc_id = self.__get_document_id(doc)
        return self._rs_index.tokens(doc_id, index, count)

    @__require_open_index
    def intervals(
        self, doc: 'CaptionIndex.DocIdOrDocument',
        start_time: float = 0.,
        end_time: float = float('inf')
    ) -> Iterable['CaptionIndex.Posting']:
        """Get time intervals in the document"""
        doc_id = self.__get_document_id(doc)
        return [
            CaptionIndex.Posting(*p)
            for p in self._rs_index.intervals(doc_id, start_time, end_time)]

    @__require_open_index
    def position(
        self, doc: 'CaptionIndex.DocIdOrDocument',
        time_offset: float
    ) -> int:
        """Find next token position containing or near the time offset"""
        doc_id = self.__get_document_id(doc)
        return self._rs_index.position(doc_id, time_offset)

    def close(self) -> None:
        self._rs_index = None

    # Internal helper methods

    def __enter__(self) -> 'CaptionIndex':
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.close()

    def __get_document_id(self, doc: 'CaptionIndex.DocIdOrDocument') -> int:
        if isinstance(doc, Documents.Document):
            return doc.id
        else:
            return self._documents[doc].id

    def __get_document_ids(
        self, docs: Optional[Iterable['CaptionIndex.DocIdOrDocument']]
    ) -> List['CaptionIndex.DocIdOrDocument']:
        return [] if docs is None else [
            self.__get_document_id(d) for d in docs]

    def __get_word_opts(
        self, word: Union[
            'CaptionIndex.WordIdOrWord',
            List['CaptionIndex.WordIdOrWord']
        ]
    ) -> List[Lexicon.Word]:
        if isinstance(word, Lexicon.Word):
            return [word]
        elif isinstance(word, (str, int)):
            return [self._lexicon[word]]
        else:
            return [w if isinstance(w, Lexicon.Word) else self._lexicon[w]
                    for w in word]

    def __get_ngram_ids_and_anchor_idx(self, words):
        ngram_word_ids = []
        min_cost = 0xFFFFFFFF
        min_cost_idx = None
        for i, word in enumerate(words):
            word_opts = self.__get_word_opts(word)
            ngram_word_ids.append([w.id for w in word_opts])
            word_cost = sum(w.count for w in word_opts)
            if word_cost < min_cost:
                min_cost = word_cost
                min_cost_idx = i
        return ngram_word_ids, min_cost_idx

    def __tokenize_text(self, text: str) -> List[str]:
        tokens = list(self.tokenizer().tokens(text.strip()))
        if len(tokens) == 0:
            raise ValueError('No tokens in input')
        return tokens

    def __unpack_rs_search(self, result) -> Generator:
        for doc_id, postings in result:
            yield CaptionIndex.Document(
                id=doc_id, postings=[
                    CaptionIndex.Posting(*p) for p in postings])


class BinaryFormat(object):
    """
    Binary data formatter for writing and reading the indexes

    Supports 4 data types:
        - u32
        - time interval
        - datum
        - byte
    """

    class Config(NamedTuple):
        start_time_bytes: int   # Number of bytes to encode start times
        end_time_bytes: int     # Number of bytes to encode end - start
        datum_bytes: int        # Number of bytes to encode other data

    def __init__(self, config: 'BinaryFormat.Config'):
        self._endian = 'little'

        assert config.start_time_bytes > 0
        assert config.end_time_bytes > 0
        self._start_time_bytes = config.start_time_bytes
        self._end_time_bytes = config.end_time_bytes

        assert config.datum_bytes > 0
        self._datum_bytes = config.datum_bytes

        # Derived values
        self._time_interval_bytes = (
            config.start_time_bytes + config.end_time_bytes)
        self._max_time_interval = (
            2 ** (8 * (config.start_time_bytes - config.end_time_bytes)) - 1)
        self._max_datum_value = 2 ** (config.datum_bytes * 8) - 1

    @property
    def u32_bytes(self) -> int:
        return 4

    @property
    def time_interval_bytes(self) -> int:
        return self._time_interval_bytes

    @property
    def datum_bytes(self) -> int:
        return self._datum_bytes

    @property
    def max_time_interval(self) -> int:
        """Largest number of milliseconds between start and end times"""
        return self._max_time_interval

    @property
    def max_datum_value(self) -> int:
        """Largest value that can be serialized"""
        return self._max_datum_value

    def encode_u32(self, data: int) -> bytes:
        assert isinstance(data, int)
        return (data).to_bytes(4, self._endian)

    def encode_time_interval(self, start: int, end: int) -> bytes:
        assert isinstance(start, int)
        assert isinstance(end, int)
        diff = end - start
        if diff < 0:
            raise ValueError(
                'start cannot exceed end: {} > {}'.format(start, end))
        if diff > self.max_time_interval:
            raise ValueError('end - start > {}'.format(self.max_time_interval))
        return (
            (start).to_bytes(self._start_time_bytes, self._endian)
            + (diff).to_bytes(self._end_time_bytes, self._endian))

    def encode_datum(self, i: int) -> bytes:
        assert isinstance(i, int)
        if i < 0:
            raise ValueError('Out of range: {} < 0'.format(i))
        if i > self._max_datum_value:
            raise ValueError('Out of range: {} > {}'.format(
                             i, self._max_datum_value))
        return (i).to_bytes(self._datum_bytes, self._endian)

    def _decode_u32(self, s: bytes) -> int:
        assert len(s) == 4, '{} is the wrong length'.format(len(s))
        return int.from_bytes(s, self._endian)

    def _decode_time_interval(self, s: bytes) -> Tuple[int, int]:
        assert len(s) == self.time_interval_bytes
        start = int.from_bytes(s[:self._start_time_bytes], self._endian)
        diff = int.from_bytes(s[self._start_time_bytes:], self._endian)
        return start, start + diff

    def _decode_datum(self, s: bytes) -> int:
        assert len(s) == self._datum_bytes, \
            '{} is the wrong length'.format(len(s))
        return int.from_bytes(s, self._endian)

    @staticmethod
    def default() -> 'BinaryFormat':
        return BinaryFormat(
            BinaryFormat.Config(
                start_time_bytes=4,
                end_time_bytes=2,
                datum_bytes=3))
