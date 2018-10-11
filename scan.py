#!/usr/bin/env python3

import argparse
import os
import time
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool

from util.index import Lexicon, Documents, DocumentData


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


DOCUMENT_DATA = None


def count_tokens(i):
    count = 0
    for t in DOCUMENT_DATA.tokens(i):
        count += 1
    return count


def main(index_dir, workers, limit):
    doc_path = os.path.join(index_dir, 'docs.list')
    data_path = os.path.join(index_dir, 'docs.bin')
    lex_path = os.path.join(index_dir, 'words.lex')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    with DocumentData(data_path, lexicon, documents) as docdata:
        global DOCUMENT_DATA
        DOCUMENT_DATA = docdata

        if limit is None:
            limit = len(documents)

        with tqdm(total=limit, desc='Counting tokens') as pbar, \
                Pool(processes=workers) as pool:

            def progress(ignored):
                pbar.update(1)

            start_time = time.time()

            results = deque()
            for i in range(limit):
                results.append(pool.apply_async(
                    count_tokens, (i,), callback=progress))

            count = 0
            for a in results:
                count += a.get()
            end_time = time.time()

        print('Scanned {} documents for {} tokens in {:d}s'.format(
            limit, count, int(end_time - start_time)))


if __name__ == '__main__':
    main(**vars(get_args()))
