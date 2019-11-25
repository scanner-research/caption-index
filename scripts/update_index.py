#!/usr/bin/env python3

"""
Update an existig index

This will produce:
 - an updated document list
 - index file(s) for the additional documents
"""

import argparse
import os
import shutil
from collections import deque
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Optional

from captions import Lexicon, Documents
from captions.indexer import index_document

from lib.common import *


# Hack to get around sharing args beween processes workers
WORKER_LEXICON = None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('new_doc_dir', type=str,
                   help='Directory containing transcripts')
    p.add_argument('index_dir', type=str,
                   help='Directory containing index')
    p.add_argument('-j', dest='parallelism', type=int,
                   default=DEFAULT_PARALLELISM,
                   help='Number of CPU cores to use. Default: {}'.format(
                        DEFAULT_PARALLELISM))
    p.add_argument('--chunk-size', dest='chunk_size', type=int,
                   help='Break the index into chunks of n documents')
    p.add_argument('--ext', dest='file_ext',
                   default=DEFAULT_SOURCE_FILE_EXT,
                   help='Subtitle file extension. Default: {}'.format(
                        DEFAULT_SOURCE_FILE_EXT))
    return p.parse_args()


def index_single_doc(doc_id: int, doc_path: str, out_path: str):
    index_document(doc_id, doc_path, WORKER_LEXICON, out_path)


def index_new_docs(
    doc_dir: str, new_documents: Documents, out_dir: str,
    chunk_size: Optional[int], parallelism: int
):
    """Builds inverted indexes and reencode documents in binary"""
    with tqdm(total=len(new_documents), desc='Building indexes') as pbar, \
            Pool(processes=parallelism) as pool:

        def progress(ignored):
            pbar.update(1)

        results = deque()
        for doc in new_documents:
            doc_path = os.path.join(doc_dir, doc.name)
            doc_out_path = os.path.join(out_dir, str(doc.id))
            results.append((
                pool.apply_async(
                    index_single_doc, (doc.id, doc_path, doc_out_path),
                    callback=progress),
                doc_out_path))

        for async_result, _ in results:
            async_result.get()

    return [x for _, x in results]


def main(
    new_doc_dir: str, index_dir: str,
    parallelism: int = DEFAULT_PARALLELISM,
    chunk_size: Optional[int] = None,
    file_ext: str = DEFAULT_SOURCE_FILE_EXT
):
    assert chunk_size is None or chunk_size > 0
    assert parallelism > 0
    doc_path = os.path.join(index_dir, 'documents.txt')
    lex_path = os.path.join(index_dir, 'lexicon.txt')
    index_path = os.path.join(index_dir, 'index.bin')

    global WORKER_LEXICON
    WORKER_LEXICON = Lexicon.load(lex_path)

    documents = Documents.load(doc_path)

    new_doc_names = list_docs(new_doc_dir, file_ext)
    assert len(new_doc_names) > 0
    for new_doc in new_doc_names:
        if new_doc in documents:
            raise Exception('{} is already indexed! Aborting.'.format(new_doc))

    base_doc_id = len(documents)
    new_documents = [Documents.Document(id=i + base_doc_id, name=d)
                     for i, d in enumerate(new_doc_names)]

    # Index the new docyments
    tmp_dir = os.path.join(index_dir, 'update-index.tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    new_doc_index_paths = index_new_docs(
        new_doc_dir, new_documents, tmp_dir, chunk_size, parallelism)

    # Convert to a dirctory
    if os.path.isfile(index_path):
        tmp_index_path = index_path + '.tmp'
        shutil.move(index_path, tmp_index_path)
        os.makedirs(index_path)
        shutil.move(
            tmp_index_path,
            os.path.join(index_path, '0-{}.bin'.format(base_doc_id)))

    assert os.path.isdir(index_path)
    if chunk_size is None:
        new_index_path = os.path.join(
            index_path, '{}-{}.bin'.format(
                base_doc_id, base_doc_id + len(new_doc_index_paths)))
        merge_index_files(new_doc_index_paths, new_index_path)
    elif chunk_size == 1:
        for fname in os.listdir(tmp_dir):
            shutil.move(os.path.join(tmp_dir, fname), index_path)
    else:
        max_doc_id = base_doc_id + len(new_doc_index_paths)
        for i in range(
            base_doc_id, max_doc_id, chunk_size
        ):
            new_index_path = os.path.join(index_path, '{}-{}.bin'.format(
                i, min(i + chunk_size, max_doc_id)))
            merge_index_files(
                new_doc_index_paths[i:i + chunk_size], new_index_path)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Write out the new documents file
    shutil.move(doc_path, doc_path + '.old')
    all_documents = [d for d in documents]
    all_documents.extend(new_documents)
    Documents(all_documents).store(doc_path)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
