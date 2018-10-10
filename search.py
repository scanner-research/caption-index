#!/usr/bin/env python3

import argparse
import os
import traceback

from util.index import Lexicon, Documents, InvertedIndex


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('query', nargs='*')
    return parser.parse_args()


def format_seconds(s):
    hours = int(s / 3600)
    minutes = int(s / 60) - hours * 60
    seconds = int(s) - hours * 3600 - minutes * 60
    millis = int((s - int(s)) * 1000)
    return '{:02d}h {:02d}m {:02d}.{:03d}s'.format(hours, minutes, seconds, millis)


def main(index_dir, query):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    def run_search(text):
        try:
            results = index.search(text)
        except Exception as e:
            traceback.print_exc()
            return

        for d in results:
            print(documents[d.id])
            for i, e in enumerate(d.entries):
                print(' {}-- [{} - {}] position={}'.format(
                    '\\' if i == len(d.entries) - 1 else '|',
                    format_seconds(e.start), format_seconds(e.end), e.position))
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
