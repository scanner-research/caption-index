#!/usr/bin/env python3

"""
Count all of the tokens in all of the documents
"""

import argparse
import os
import time
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool

from captions import Lexicon, Documents


DEFAULT_WORKERS = os.cpu_count()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('index_dir', type=str,
                   help='Directory containing index files')
    p.add_argument('-j', dest='workers', type=int, default=DEFAULT_WORKERS,
                   help='Number of CPU cores to use. Default: {}'.format(
                        DEFAULT_WORKERS))
    p.add_argument('--limit', dest='limit', type=int,
                   help='Limit the number of documents to scan')
    return p.parse_args()


def count_tokens(i):
    """Do an expensive linear scan of all of the tokens"""
    count = 0
    for t in count_tokens.documents.open(i).tokens():
        count += 1
    return count


def init_worker(function, index_dir):
    doc_path = os.path.join(index_dir, 'documents.txt')
    data_dir = os.path.join(index_dir, 'data')
    function.documents = Documents.load(doc_path)
    function.documents.configure(data_dir)


def main(index_dir, workers, limit):
    doc_path = os.path.join(index_dir, 'documents.txt')
    documents = Documents.load(doc_path)

    if limit is None:
        limit = len(documents)

    start_time = time.time()
    with Pool(processes=workers, initializer=init_worker,
              initargs=(count_tokens, index_dir)) as pool:
        count = 0
        for n in tqdm(pool.imap_unordered(count_tokens, range(limit)),
                      desc='Counting tokens'):
            count += n

    print('Scanned {} documents for {} tokens in {:d}ms'.format(
        limit, count, int(1000 * (time.time() - start_time))))


if __name__ == '__main__':
    main(**vars(get_args()))
