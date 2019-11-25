#!/usr/bin/env python3

"""
Build a metadata index containing part-of-speech tags.

You will need spacy:
    python3 -m spacy download en
"""

import argparse
import os
import spacy
import traceback
from spacy.tokens import Doc
from tqdm import tqdm

from captions import Lexicon, Documents, CaptionIndex
from captions.metadata import MetadataFormat


NLP = spacy.load('en', disable=['ner', 'parser'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                        help='Fix casing of documents')
    return parser.parse_args()


POS_TAGS = [
    '""', '#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'BES',
    'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN', 'JJ', 'JJR',
    'JJS', 'LS', 'MD', 'NFP', 'NIL', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
    'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SP', 'SYM', 'TO', 'UH', 'VB',
    'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP',
    '``'
]
POS_TAGS_MAP = {t: i for i, t in enumerate(POS_TAGS)}
POS_UNKNOWN_TAG = 'XX'
POS_UNKNOWN_ID = POS_TAGS_MAP[POS_UNKNOWN_TAG]


class POSTag(object):

    @staticmethod
    def encode(s):
        assert isinstance(s, str)
        return POS_TAGS_MAP.get(s, POS_UNKNOWN_ID)

    @staticmethod
    def decode(b):
        assert isinstance(b, bytes)
        i = b[0]
        if i < 0 or i >= len(POS_TAGS):
            return 'XX'     # unknown
        return POS_TAGS[i]


class NLPTagFormat(MetadataFormat):

    @property
    def size(self):
        return 1

    def decode(self, i):
        return POSTag.decode(i)


def write_doc_metadata(d, tokens, out_file):
    if len(tokens) > 0:
        try:
            tags = [t.tag_ for t in NLP.tagger(Doc(NLP.vocab, words=tokens))]
        except Exception:
            traceback.print_exc()
            print('Failed to generate tags: {}'.format(d.name))
            tags = [POS_UNKNOWN_TAG] * len(tokens)
        out_file.write(MetadataFormat.header(d.id, len(tags)))
        out_file.write(bytes(POSTag.encode(t) for t in tags))


def main(index_dir, lowercase):
    index_path = os.path.join(index_dir, 'index.bin')
    doc_path = os.path.join(index_dir, 'documents.txt')
    lex_path = os.path.join(index_dir, 'lexicon.txt')
    meta_path = os.path.join(index_dir, 'meta.bin')

    if os.path.exists(meta_path):
        raise Exception('Metadata binary already exists!')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    with CaptionIndex(index_path, lexicon, documents) as index, \
            open(meta_path, 'wb') as f:
        for d in tqdm(documents, desc='Writing metadata'):
            doc_tokens = [lexicon.decode(t) for t in index.tokens(d)]
            if lowercase:
                doc_tokens = [t.lower() for t in doc_tokens]
            write_doc_metadata(d, doc_tokens, f)


if __name__ == '__main__':
    main(**vars(get_args()))
