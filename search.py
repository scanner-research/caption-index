#!/usr/bin/env python3

import argparse
import os
import traceback

from util.index import Lexicon, Documents, InvertedIndex, tokenize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('query', nargs='*')
    return parser.parse_args()


def main(index_dir, query):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    def run_search(text):
        try:
            results = index.unigram_search(text)
        except Exception as e:
            traceback.print_exc()
            return

        for d in results:
            print(documents[d.id])
            for i, e in enumerate(d.entries):
                print(' {}-- [{}s - {}s] position={}'.format(
                    '\\' if i == len(d.entries) - 1 else '|',
                    e.start, e.end, e.position))
        print('Found {} documents, {} occurences'.format(
            len(results), sum(len(d.entries) for d in results)))

    with InvertedIndex(idx_path, lexicon, documents) as index:
        if len(query) > 0:
            print('Query: ', query)
            run_search(' '.join(query))
        else:
            while True:
                try:
                    query = input('> ')
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                query = query.strip()
                if len(query) > 0:
                    run_search(query)


if __name__ == '__main__':
    main(**vars(get_args()))
