from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):

    @abstractmethod
    def tokens(self, s: str) -> List[str]:
        pass


class SpacyTokenizer(Tokenizer):

    def __init__(self):
        # Lazy import
        import spacy
        self._tokenizer = spacy.load('en', disable=['tagger', 'parser', 'ner'])

    def tokens(self, text: str) -> List[str]:
        return [t.text for t in self._tokenizer(text)]


_DEFAULT_TOKENIZER = None


def default_tokenizer():
    """Get the default tokenizer"""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = SpacyTokenizer()
    return _DEFAULT_TOKENIZER
