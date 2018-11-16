#!/usr/bin/env python3

"""
Find lexicons of words around phrases
"""

import argparse
import os
import spacy
import time
import traceback

from captions import tokenize, Lexicon, Documents, InvertedIndex, \
    DocumentData, NgramFrequency
from captions.util import pmi_search


DEFAULT_WINDOW = 30
DEFAULT_N = 5
DEFAULT_RESULT_LEN = 25


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('-w', dest='window_size', type=int,
                        default=DEFAULT_WINDOW,
                        help='Window seconds (default: {})'.format(
                             DEFAULT_WINDOW))
    parser.add_argument('-n', dest='n', type=int, default=DEFAULT_N,
                        help='Max n-gram size (default: {})'.format(DEFAULT_N))
    parser.add_argument('-l', dest='result_len', type=int,
                        default=DEFAULT_RESULT_LEN,
                        help='Number of results to print (default: {})'.format(
                              DEFAULT_RESULT_LEN))
    parser.add_argument('query', nargs='*')
    return parser.parse_args()


NLP = spacy.load('en')
BLACKLIST_TOKENS = {'\n', '.', ',', '?', '!', '>', ':', '[', ']', '"', '(', ')'}


def filter_results(tokens):
    for t in tokens:
        if t in BLACKLIST_TOKENS:
            return False
        t_lower = t.lower()
        if NLP.vocab[t_lower].is_stop:
            return False
    return True


def run_search(query, lexicon, index, document_data, ngram_frequency, n,
               window_size, result_len):
    start_time = time.time()
    query_list = []
    for q in query.split(','):
        q = q.strip()
        if len(q) > 0:
            try:
                query_list.append(tokenize(q))
            except KeyError:
                print('Not found:', q)
    result = pmi_search(query_list, index, document_data, ngram_frequency, n,
                        window_size)
    elapsed_time = time.time() - start_time

    for i in range(1, n + 1):
        selected = []
        for ngram, score in result.most_common():
            if len(ngram) == i:
                tokens = tuple(lexicon[t].token for t in ngram)
                if filter_results(tokens):
                    selected.append((' '.join(tokens), score))
                    if len(selected) == result_len:
                        break
        print('Top {}-grams:'.format(i))
        for i, (ngram_str, score) in enumerate(selected):
            print(' {}-- {:0.3f}\t{}'.format(
                  '\\' if i == len(selected) - 1 else '|', score, ngram_str))

    print('Completed in {:d}ms'.format(int(elapsed_time * 1000)))


def main(index_dir, query, n, window_size, result_len):
    doc_path = os.path.join(index_dir, 'docs.list')
    data_path = os.path.join(index_dir, 'docs.bin')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')
    ngram_path = os.path.join(index_dir, 'ngrams.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)
    ngram_frequency = NgramFrequency(ngram_path, lexicon)

    with InvertedIndex(idx_path, lexicon, documents) as index, \
            DocumentData(data_path, lexicon, documents) as document_data:
        if len(query) > 0:
            print('Query: ', query)
            run_search(' '.join(query), lexicon, index, document_data,
                       ngram_frequency, n, window_size, result_len)
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
                        run_search(query, lexicon, index, document_data,
                                   ngram_frequency, n, window_size, result_len)
                    except Exception:
                        traceback.print_exc()


if __name__ == '__main__':
    main(**vars(get_args()))
