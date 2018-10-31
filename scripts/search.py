#!/usr/bin/env python3

"""
Search for phrases across all of the documents
"""

import argparse
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../captions')

from index import tokenize, Lexicon, Documents, InvertedIndex, DocumentData
from util import topic_search


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


def run_search(query, documents, index, document_data, context_size, silent):
    start_time = time.time()

    if ',' not in query:
        # Phrase search
        query_tokens = list(tokenize(query))
        result = index.search(query_tokens)
    else:
        # Topic search
        query_list = []
        for q in query.split(','):
            q = q.strip()
            if len(q) > 0:
                try:
                    query_list.append(tokenize(q))
                except KeyError:
                    print('Not found:', q)
        result = topic_search(query_list, index)

    total_seconds = 0
    occurence_count = 0
    i = 0
    for i, d in enumerate(result.documents):
        if not silent:
            print(documents[d.id].name)
        occurence_count += d.count
        for j, e in enumerate(d.locations):
            total_seconds += e.end - e.start
            if not silent:
                if context_size > 0:
                    context = ' '.join(document_data.tokens(
                        d.id,
                        start_pos=max(e.min_index - context_size, 0),
                        end_pos=e.max_index + context_size,
                        decode=True
                    ))
                else:
                    context = query

                print(' {}-- [{} - {}] [position: {}] "{}"'.format(
                    '\\' if j == d.count - 1 else '|',
                    format_seconds(e.start), format_seconds(e.end),
                    e.min_index
                        if e.min_index == e.max_index
                        else '{}-{}'.format(e.min_index, e.max_index),
                    context))

    if result.count is not None:
        assert result.count == i + 1
    print('Found {} documents, {} occurences, spanning {:d}s in {:d}ms'.format(
          i, occurence_count, int(total_seconds),
          int((time.time() - start_time) * 1000)))


def main(index_dir, query, silent, context_size):
    doc_path = os.path.join(index_dir, 'docs.list')
    data_path = os.path.join(index_dir, 'docs.bin')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    with InvertedIndex(idx_path, lexicon, documents) as index, \
            DocumentData(data_path, lexicon, documents) as document_data:
        if len(query) > 0:
            print('Query: ', query)
            run_search(' '.join(query), documents, index, document_data,
                       context_size, silent)
        else:
            print('Enter a phrase or topic (phrases separated by commas):')
            while True:
                try:
                    query = input('> ')
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                query = query.strip()
                if len(query) > 0:
                    try:
                        run_search(query, documents, index, document_data,
                                   context_size, silent)
                    except Exception:
                        traceback.print_exc()


if __name__ == '__main__':
    main(**vars(get_args()))
