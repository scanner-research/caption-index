#!/usr/bin/env python3

"""
Search for phrases across all of the documents
"""

import argparse
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')

from index import Lexicon, Documents, InvertedIndex, DocumentData, tokenize


DEFAULT_CONTEXT = 3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('-s', dest='silent', action='store_true',
                        help='Silent mode for benchmarking')
    parser.add_argument('-c', dest='context_size', type=int,
                        default=DEFAULT_CONTEXT,
                        help='Context window width (default: {})'.format(
                             DEFAULT_CONTEXT))
    parser.add_argument('query', nargs='*')
    return parser.parse_args()


def format_seconds(s):
    hours = int(s / 3600)
    minutes = int(s / 60) - hours * 60
    seconds = int(s) - hours * 3600 - minutes * 60
    millis = int((s - int(s)) * 1000)
    return '{:02d}h {:02d}m {:02d}.{:03d}s'.format(
        hours, minutes, seconds, millis)


def main(index_dir, query, silent, context_size):
    doc_path = os.path.join(index_dir, 'docs.list')
    data_path = os.path.join(index_dir, 'docs.bin')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    def run_search(query):
        start_time = time.time()
        query_tokens = tokenize(query)
        try:
            result = index.search(query_tokens)
        except Exception as e:
            traceback.print_exc()
            return

        occurence_count = 0
        i = 0
        for i, d in enumerate(result.documents):
            if not silent:
                print(documents[d.id].name)
            occurence_count += d.count
            for j, e in enumerate(d.locations):
                if not silent:
                    if context_size > 0:
                        context = ' '.join(t for t, _ in docdata.tokens(
                            d.id,
                            start_pos=max(e.index - context_size, 0),
                            end_pos=e.index + context_size + len(query_tokens),
                            decode=True
                        ))
                    else:
                        context = ' '.join(query_tokens)

                    print(' {}-- [{} - {}] [index: {}] "{}"'.format(
                        '\\' if j == d.count - 1 else '|',
                        format_seconds(e.start), format_seconds(e.end),
                        e.index, context))

        if result.count is not None:
            assert result.count == i + 1
        print('Found {} documents, {} occurences in {:d}ms'.format(
              i, occurence_count, int((time.time() - start_time) * 1000)))

    with InvertedIndex(idx_path, lexicon, documents) as index, \
            DocumentData(data_path, lexicon, documents) as docdata:
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
