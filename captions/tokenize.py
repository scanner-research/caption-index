import string
import re
from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):

    @abstractmethod
    def tokens(self, s: str) -> List[str]:
        pass


def _sanitize(t: str) -> str:
    return ''.join(filter(lambda x: x in string.printable, t)).strip()


class SpacyTokenizer(Tokenizer):

    def __init__(self):
        # Lazy import
        import spacy
        self._tokenizer = spacy.load('en', disable=['tagger', 'parser', 'ner'])

    def tokens(self, text: str) -> List[str]:
        tokens = (_sanitize(t.text) for t in self._tokenizer(text))
        return [t for t in tokens if t]


class BasicTokenizer(Tokenizer):

    def __init__(self):
        self._re = re.compile(r'[.,!?:;(){}[\]`|"\']')

    def tokens(self, text: str) -> List[str]:
        result = []
        for t in _sanitize(text).split():
            if len(t) == 0:
                continue
            i = 0
            for m in self._re.finditer(t):
                token = t[i:m.start(0)]
                if token:
                    result.append(token)
                result.append(m.group(0))
                i = m.end(0)
            remainder = t[i:]
            if remainder:
                result.append(remainder)
        return result

    
class AlignmentTokenizer(Tokenizer):
    def __init__(self):
        self._tokenizer = default_tokenizer()
        
    def tokens(self, t):
        if t[0] == '{' and t[-1] == '}':
            t = t[1:-1]
        return self._tokenizer.tokens(t)


_DEFAULT_TOKENIZER = None


def default_tokenizer():
    """Get the default tokenizer"""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = BasicTokenizer()
    return _DEFAULT_TOKENIZER
