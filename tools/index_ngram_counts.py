#!/usr/bin/env python3

"""
Count ngrams in all of the documents and writes
"""

import argparse
import os
import time
import pickle
from tqdm import tqdm
from collections import deque, Counter
from multiprocessing import Pool

from captions import Lexicon, Documents, CaptionIndex
from captions.util import window

DEFAULT_WORKERS = os.cpu_count()
DEFAULT_N = 3
DEFAULT_THRESH = 1000


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('index_dir', type=str,
                   help='Directory containing index files')
    p.add_argument('-n', dest='n', type=int, default=DEFAULT_N,
                   help='Largest n-gram size. Default: {}'.format(DEFAULT_N))
    p.add_argument('--min-count', dest='min_count', type=int,
                   default=DEFAULT_THRESH,
                   help='Minimum count for ngrams. Default: {}'.format(
                        DEFAULT_THRESH))
    p.add_argument('-j', dest='workers', type=int, default=DEFAULT_WORKERS,
                   help='Number of CPU cores to use. Default: {}'.format(
                        DEFAULT_WORKERS))
    p.add_argument('--limit', dest='limit', type=int,
                   help='Limit the number of documents to scan')
    return p.parse_args()


LEXICON = None
INDEX = None
NGRAM_COUNTS = None


def filter_lexicon(lexicon, min_count):
    counts = {}
    total = 0
    for w in lexicon:
        total += w.count
        if w.count > min_count:
            counts[(w.id,)] = w.count
    return counts, total


def batch_count_ngrams(base_doc, batch_size, n):
    counts = Counter()
    total = 0
    for i in range(base_doc, base_doc + batch_size):
        for t in window(INDEX.tokens(i), n):
            total += 1
            if t[1:] in NGRAM_COUNTS and t[1-n:] in NGRAM_COUNTS:
                counts[t] += 1
    return batch_size, counts, total


def single_pass_count_ngrams(n, limit, workers, batch_size=1000):
    with tqdm(total=limit, desc='Counting {}-grams'.format(n)) as pbar, \
            Pool(processes=workers) as pool:

        def progress(ignored):
            pbar.update(ignored[0])

        start_time = time.time()

        results = deque()
        for i in range(0, limit, batch_size):
            results.append(pool.apply_async(
                batch_count_ngrams,
                (i, min(batch_size, limit - i), n), callback=progress))

        counts = Counter()
        total = 0
        for a in results:
            _, doc_counts, doc_total = a.get()
            counts += doc_counts
            total += doc_total
        end_time = time.time()

    print('Scanned {} documents for {}-grams ({} total, {} unique) in {:d}s'.format(
          limit, n, total, len(counts), int(end_time - start_time)))
    return counts, total


def main(index_dir, n, min_count, workers, limit):
    index_path = os.path.join(index_dir, 'index.bin')
    doc_path = os.path.join(index_dir, 'documents.txt')
    lex_path = os.path.join(index_dir, 'lexicon.txt')
    out_path = os.path.join(index_dir, 'ngrams.bin')

    if os.path.exists(out_path):
        raise FileExistsError('{} already exists'.format(out_path))

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    global INDEX, NGRAM_COUNTS
    NGRAM_COUNTS, unigram_total = filter_lexicon(lexicon, min_count)
    ngram_total = [unigram_total]

    with CaptionIndex(index_path, lexicon, documents) as index:
        INDEX = index

        if limit is None:
            limit = len(documents)

        for i in range(2, n + 1):
            counts, total = single_pass_count_ngrams(i, limit, workers)
            for ngram in counts:
                count = counts[ngram]
                if count >= min_count:
                    NGRAM_COUNTS[ngram] = count
            ngram_total.append(total)
            del counts, total

    print('Saving results: {} ngrams'.format(len(NGRAM_COUNTS)))
    with open(out_path, 'wb') as f:
        pickle.dump([NGRAM_COUNTS, ngram_total], f)


if __name__ == '__main__':
    main(**vars(get_args()))
