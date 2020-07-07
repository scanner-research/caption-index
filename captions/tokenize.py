from abc import ABC, abstractmethod
from typing import List

from .rs_captions import tokenize


class Tokenizer(ABC):

    @abstractmethod
    def tokens(self, text: str) -> List[str]:
        pass


class BasicTokenizer(Tokenizer):

    def tokens(self, text: str) -> List[str]:
        return tokenize(text)


class AlignmentTokenizer(Tokenizer):
    """Our aligned caption files have {}s around misaligned words."""

    def __init__(self):
        self._tokenizer = default_tokenizer()

    def tokens(self, text: str) -> List[str]:
        if text[0] == '{' and text[-1] == '}':
            text = text[1:-1]
        return tokenize(text)


_DEFAULT_TOKENIZER = None


def default_tokenizer():
    """Get the default tokenizer"""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = BasicTokenizer()
    return _DEFAULT_TOKENIZER
