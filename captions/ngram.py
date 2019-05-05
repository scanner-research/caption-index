import pickle
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union, Tuple, Iterable, Optional

from .tokenize import default_tokenizer, Tokenizer
from .index import Documents, Lexicon


class NgramFrequency(object):
    """
    A map from ngrams to their frequencies

    Get the frequency of an ngram of all other ngrams of that length
    with the following:
        freq = ngram_frequency[('hello', 'there')]

    Test if a ngram exists:
        if ('hello', 'world') in ngram_frequency:
            ...
    """

    class NgramDoesNotExist(Exception):
        pass

    Ngram = Union[str, Tuple]

    def __init__(self, path: str, lexicon: Lexicon,
                 tokenizer: Optional[Tokenizer] = None):
        """Dictionary of ngram to frequency"""
        assert isinstance(path, str)
        assert isinstance(lexicon, Lexicon)
        self._lexicon = lexicon

        if tokenizer is None:
            tokenizer = default_tokenizer()
        self._tokenizer = tokenizer

        with open(path, 'rb') as f:
            self._counts, self._totals = pickle.load(f)

    def __iter__(self) -> Iterable['NgramFrequency.Ngram']:
        return self._counts.__iter__()

    def __getitem__(self, key: 'NgramFrequency.Ngram') -> float:
        if isinstance(key, str):
            key = tuple(self._tokenizer.tokens(key.strip()))
        denom = self._totals[len(key) - 1]
        if isinstance(key[0], int):
            try:
                return self._counts[key] / denom
            except KeyError:
                raise NgramFrequency.NgramDoesNotExist(repr(key))
        elif isinstance(key[0], str):
            key = tuple(self._lexicon[k].id for k in key)
            try:
                return self._counts[key] / denom
            except KeyError:
                raise NgramFrequency.NgramDoesNotExist(repr(key))
        raise TypeError('Not supported for {}'.format(type(key)))

    def __contains__(self, key: 'NgramFrequency.Ngram') -> bool:
        try:
            self.__getitem__(key)
        except NgramFrequency.NgramDoesNotExist:
            return False
        return True

    def __len__(self) -> int:
        return len(self._counts)
