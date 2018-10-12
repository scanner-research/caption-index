"""
NLP utils
"""

import spacy


MODEL = 'en'
_TOKENIZER = spacy.load(MODEL, disable=['tagger', 'parser', 'ner'])


def tokenize(text):
    return (t.text for t in _TOKENIZER(text))
