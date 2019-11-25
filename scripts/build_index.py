#!/usr/bin/env python3

"""
Index a directory of transcript files.

This will produce:
 - document list
 - a lexicon
 - index (one or more files depending on chunk size)
"""

import argparse
import os
import shutil
from collections import deque, Counter
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Optional

from captions import Lexicon, Documents
from captions.indexer import index_document, get_document_word_counts

from lib.common import *

DEFAULT_OUT_DIR = 'out'

# Hack to get around sharing args beween processes workers
WORKER_LEXICON = None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('doc_dir', type=str,
                   help='Directory containing transcripts')
    p.add_argument('-o', dest='out_dir', type=str, default=DEFAULT_OUT_DIR,
                   help='Output directory. Default: {}'.format(
                        DEFAULT_OUT_DIR))
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
    p.add_argument('--keep-tmp-files', dest='keep_tmp_files',
                   action='store_true', help='Keeps per document index files')
    return p.parse_args()


def get_doc_word_counts(doc_path: str) -> Counter:
    return get_document_word_counts(doc_path, max_word_len=MAX_WORD_LEN)


def get_word_counts(doc_dir: str, doc_names: List[str], parallelism: int):
    words = Counter()
    with tqdm(total=len(doc_names), desc='Building lexicon') as pbar, \
            Pool(processes=parallelism) as pool:

        def collect(result):
            pbar.update(1)

        async_results = deque()
        for d in doc_names:
            doc_path = os.path.join(doc_dir, d)
            async_results.append(pool.apply_async(
                get_doc_word_counts, (doc_path,), callback=collect))

        # Forces exceptions to be rethrown
        for a in async_results:
            words.update(a.get())

    print('Lexicon size: {}'.format(len(words)))
    return words


def index_single_doc(doc_id: int, doc_path: str, out_path: str):
    index_document(doc_id, doc_path, WORKER_LEXICON, out_path)


def index_all_docs(
    doc_dir: str, documents: Documents, lexicon: Lexicon,
    out_path: str, tmp_dir: str, chunk_size: Optional[int],
    parallelism: int, keep_tmp_files: bool
):
    """Builds inverted indexes and reencode documents in binary"""
    global WORKER_LEXICON
    WORKER_LEXICON = lexicon

    with tqdm(total=len(documents), desc='Building indexes') as pbar, \
            Pool(processes=parallelism) as pool:

        def progress(ignored):
            pbar.update(1)

        results = deque()
        for doc in documents:
            doc_path = os.path.join(doc_dir, doc.name)
            doc_out_path = os.path.join(tmp_dir, str(doc.id))
            results.append((
                pool.apply_async(
                    index_single_doc, (doc.id, doc_path, doc_out_path),
                    callback=progress),
                doc_out_path))

        for async_result, _ in results:
            async_result.get()

        # Cat the files together (in batches to avoid too many args)
        all_doc_index_paths = [x for _, x in results]

        if chunk_size is None:
            merge_index_files(
                all_doc_index_paths, out_path, keep_tmp_files=keep_tmp_files)
        elif chunk_size == 1:
            shutil.move(tmp_dir, out_path)
        else:
            os.makedirs(out_path)
            for i in range(0, len(all_doc_index_paths), chunk_size):
                out_file = os.path.join(
                    out_path, '{}-{}.bin'.format(
                        i, min(i + chunk_size, len(all_doc_index_paths))))
                merge_index_files(
                    all_doc_index_paths[i:i + chunk_size], out_file,
                    keep_tmp_files=keep_tmp_files)


def build_or_load_lexicon(
    doc_dir, doc_names, lex_path: str, parallelism: int
) -> Lexicon:
    if os.path.exists(lex_path):
        print('Loading lexicon: {}'.format(lex_path))
        lexicon = Lexicon.load(lex_path)
    else:
        print('Building lexicon: {}'.format(lex_path))
        word_counts = get_word_counts(doc_dir, doc_names, parallelism)
        lexicon = Lexicon([
            Lexicon.Word(i, w, word_counts[w])
            for i, w in enumerate(sorted(word_counts.keys()))
        ])
        print('Storing lexicon: {}'.format(lex_path))
        lexicon.store(lex_path)
    return lexicon


def main(
    doc_dir: str, out_dir: str,
    parallelism: int = DEFAULT_PARALLELISM,
    chunk_size: Optional[int] = None,
    file_ext: str = DEFAULT_SOURCE_FILE_EXT,
    keep_tmp_files: bool = False
):
    assert chunk_size is None or chunk_size > 0
    assert parallelism > 0

    # Load document names
    doc_names = list(sorted(list_docs(doc_dir, file_ext)))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Load or build a lexicon
    lex_path = os.path.join(out_dir, 'lexicon.txt')
    lexicon = build_or_load_lexicon(doc_dir, doc_names, lex_path, parallelism)
    assert os.path.exists(lex_path), 'Missing: {}'.format(lex_path)

    # Build and store the document list
    docs_path = os.path.join(out_dir, 'documents.txt')
    documents = Documents([
        Documents.Document(id=i, name=d) for i, d in enumerate(doc_names)
    ])
    print('Storing document list: {}'.format(docs_path))
    documents.store(docs_path)
    assert os.path.exists(docs_path), 'Missing: {}'.format(docs_path)

    # Build inverted index chunks and reencode the documents
    tmp_dir = os.path.join(out_dir, 'index.tmp')
    index_path = os.path.join(out_dir, 'index.bin')
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    os.makedirs(tmp_dir)
    try:
        index_all_docs(doc_dir, documents, lexicon, index_path, tmp_dir,
                       chunk_size, parallelism, keep_tmp_files)
    finally:
        if not keep_tmp_files and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    assert os.path.exists(index_path), 'Missing: {}'.format(index_path)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
