#!/usr/bin/env python3

"""
Search for phrases across all of the documents
"""

import argparse
import os
import time
import traceback

from captions import tokenize, Lexicon, Documents, CaptionIndex
from captions.util import topic_search


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


def run_search(query, documents, lexicon, index, context_size, silent):
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
    doc_count = 0
    for i, d in enumerate(result):
        if not silent:
            print(documents[d.id].name)
        occurence_count += len(d.postings)
        for j, p in enumerate(d.postings):
            total_seconds += p.end - p.start
            if not silent:
                if context_size > 0:
                    start_idx = max(p.idx - context_size, 0)
                    context = ' '.join([
                        lexicon.decode(t)
                        for t in index.tokens(
                            d.id, index=start_idx,
                            count=p.idx + p.len + context_size - start_idx)
                    ])
                else:
                    context = query

                print(' {}-- [{} - {}] [position: {}] "{}"'.format(
                    '\\' if j == len(d.postings) - 1 else '|',
                    format_seconds(p.start), format_seconds(p.end),
                    p.idx if p.len == 1 else '{}-{}'.format(p.idx, p.idx + p.len),
                    context))
        doc_count += 1
    print('Found {} documents, {} occurences, spanning {:d}s in {:d}ms'.format(
          doc_count, occurence_count, int(total_seconds),
          int((time.time() - start_time) * 1000)))


def main(index_dir, query, silent, context_size):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    with CaptionIndex(idx_path, lexicon, documents) as index:
        if len(query) > 0:
            print('Query: ', query)
            run_search(' '.join(query), documents, lexicon, index,
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
                        run_search(query, documents, lexicon, index,
                                   context_size, silent)
                    except Exception:
                        traceback.print_exc()


if __name__ == '__main__':
    main(**vars(get_args()))
