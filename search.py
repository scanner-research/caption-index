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
    return '{:02d}h {:02d}m {:02d}.{:03d}s'.format(
        hours, minutes, seconds, millis)


def main(index_dir, query):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    def run_search(text):
        try:
            result = index.search(text)
        except Exception as e:
            traceback.print_exc()
            return

        occurence_count = 0
        i = 0
        for i, d in enumerate(result.documents):
            print(documents[d.id])
            occurence_count += d.count
            for j, e in enumerate(d.locations):
                print(' {}-- [{} - {}] index={}'.format(
                    '\\' if j == d.count - 1 else '|',
                    format_seconds(e.start), format_seconds(e.end),
                    e.index))

        if result.count is not None:
            assert result.count == i + 1
        print('Found {} documents, {} occurences'.format(i, occurence_count))

    with InvertedIndex(idx_path, lexicon, documents) as index:
        if len(query) > 0:
            print('Query: ', query)
            run_search(' '.join(query))
        else:
            print('Enter a token or phrase:')
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
