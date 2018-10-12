"""
NLP utils
"""

import spacy


_NLP = None
_TOKENIZER = None


MODEL = 'en' #'en_core_web_sm'


def _load_spacy():
    global _NLP, _TOKENIZER
    _NLP = spacy.load(MODEL, disable=['parser', 'ner'])
    _TOKENIZER = spacy.load(MODEL, disable=['tagger', 'parser', 'ner'])


def tokenize(text):
    if _NLP is None:
        _load_spacy()
    return (t.text for t in _TOKENIZER(text))


def tokenize_and_tag(text):
    if _NLP is None:
        _load_spacy()
    return _NLP(text)


_POS_TAGS = [
    '""', '#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'BES',
    'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN', 'JJ', 'JJR',
    'JJS', 'LS', 'MD', 'NFP', 'NIL', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
    'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SP', 'SYM', 'TO', 'UH', 'VB',
    'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP',
    '``'
]
_POS_TAGS_MAP = {t: i for i, t in enumerate(_POS_TAGS)}
_POS_UNKNOWN_ID = _POS_TAGS_MAP['XX']


class POSTag(object):

    @staticmethod
    def encode(s):
        assert isinstance(s, str)
        return _POS_TAGS_MAP.get(s, _POS_UNKNOWN_ID)

    @staticmethod
    def decode(i):
        assert isinstance(i, int)
        if i < 0 or i >= len(_POS_TAGS):
            return 'XX'     # unknown
        return _POS_TAGS[i]
