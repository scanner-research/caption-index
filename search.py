#!/usr/bin/env python3

import argparse
import os

from util.index import Lexicon, Documents, InvertedIndex, tokenize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('-l', '--limit', type=int, required=False,
                        help='Limit number of documents printed')
    parser.add_argument('query', nargs='+')
    return parser.parse_args()


def main(index_dir, query, limit):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    print('Query: ', query)
    with InvertedIndex(idx_path, lexicon, documents) as index:
        results = index.unigram_search(query[0])
        assert len(results) > 0


if __name__ == '__main__':
    main(**vars(get_args()))
