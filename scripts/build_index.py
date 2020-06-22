#!/usr/bin/env python3

"""
Index a directory of transcript files.

This will produce:
 - document list
 - a lexicon
 - index (one or more files depending on chunk size)
 - intervals and tokens (in binary format)
"""

import argparse
import os
import shutil
from multiprocessing import Pool
from typing import List, Optional
from tqdm import tqdm

from captions import Lexicon, Documents
from captions.indexer import index_document
from captions.tokenize import AlignmentTokenizer

from lib.common import *

DEFAULT_OUT_DIR = 'out'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--doc-dir', type=str,
                   help='Directory containing captions. If not passed, read from stdin.')
    p.add_argument('-o', dest='out_dir', type=str, default=DEFAULT_OUT_DIR,
                   help='Output directory. Default: {}'.format(DEFAULT_OUT_DIR))
    p.add_argument('-j', dest='parallelism', type=int, default=DEFAULT_PARALLELISM,
                   help='Number of CPU cores to use. Default: {}'.format(DEFAULT_PARALLELISM))
    p.add_argument('--chunk-size', dest='chunk_size', type=int,
                   help='Break the index into chunks of n documents')
    p.add_argument('--keep-tmp-files', dest='keep_tmp_files',
                   action='store_true', help='Keeps per document index files')
    return p.parse_args()


def init_index_worker(function, lexicon_path: str):
    function.lexicon = Lexicon.load(lexicon_path)


def index_single_doc(args):
    doc_id, doc_path, index_path, data_path = args
    index_document(doc_id, doc_path, index_single_doc.lexicon,
                   index_path, data_path,
                   tokenizer=AlignmentTokenizer())
index_single_doc.lexicon = None


def index_all_docs(
        docs_to_index: List[DocumentToIndex],
        documents: Documents,
        lexicon_path: Lexicon,
        index_out_path: str,
        data_out_dir: str,
        tmp_dir: str,
        chunk_size: Optional[int],
        parallelism: int,
        keep_tmp_files: bool
):
    """Builds inverted indexes and reencode documents in binary"""
    assert len(docs_to_index) == len(documents)

    with Pool(
            processes=parallelism, initializer=init_index_worker,
            initargs=(index_single_doc, lexicon_path)
    ) as pool:
        worker_args = []
        doc_index_paths = []
        for doc_to_index, doc in zip(docs_to_index, documents):
            assert doc_to_index.name == doc.name
            doc_index_out_path = os.path.join(
                tmp_dir, '{:07d}.bin'.format(doc.id))
            doc_data_out_path = os.path.join(
                data_out_dir, '{}.bin'.format(doc.id))
            worker_args.append((doc.id, doc_to_index.path, doc_index_out_path,
                                doc_data_out_path))
            doc_index_paths.append(doc_index_out_path)

        for _ in tqdm(pool.imap_unordered(index_single_doc, worker_args),
                      desc='Indexing', total=len(worker_args)):
            pass

        if chunk_size is None:
            merge_files(doc_index_paths, index_out_path,
                        keep_tmp_files=keep_tmp_files)
        else:
            os.makedirs(index_out_path)
            for i in range(0, len(doc_index_paths), chunk_size):
                out_file = '{:07d}-{:07d}.bin'.format(
                    i, min(i + chunk_size, len(doc_index_paths)))
                merge_files(doc_index_paths[i:i + chunk_size],
                            os.path.join(index_out_path, out_file),
                            keep_tmp_files=keep_tmp_files)


def build_lexicon(
        docs_to_index: List[DocumentToIndex],
        lex_path: str,
        parallelism: int
) -> Lexicon:
    print('Building lexicon: {}'.format(lex_path))
    word_counts = get_word_counts(docs_to_index, parallelism)
    lexicon = Lexicon([
        Lexicon.Word(i, w, word_counts[w])
        for i, w in enumerate(sorted(word_counts.keys()))
    ])
    print('Storing lexicon: {}'.format(lex_path))
    lexicon.store(lex_path)


def remove_if_exists(fpath):
    if os.path.isdir(fpath):
        shutil.rmtree(fpath)
    elif os.path.isfile(fpath):
        os.remove(fpath)


def main(
        out_dir: str, doc_dir: Optional[str],
        parallelism: int = DEFAULT_PARALLELISM,
        chunk_size: Optional[int] = None,
        keep_tmp_files: bool = False
):
    assert chunk_size is None or chunk_size > 0
    assert parallelism > 0

    # Load document names
    if doc_dir:
        docs_to_index = list(sorted(list_docs(doc_dir)))
    else:
        docs_to_index = read_docs_from_stdin()

    os.makedirs(out_dir, exist_ok=True)

    # Load or build a lexicon
    lex_path = os.path.join(out_dir, 'lexicon.txt')
    if not os.path.exists(lex_path):
        build_lexicon(docs_to_index, lex_path, parallelism)
    assert os.path.exists(lex_path), 'Missing: {}'.format(lex_path)

    # Build and store the document list
    docs_path = os.path.join(out_dir, 'documents.txt')
    documents = Documents([
        Documents.Document(id=i, name=d.name)
        for i, d in enumerate(docs_to_index)
    ])
    print('Storing document list: {}'.format(docs_path))
    documents.store(docs_path)
    assert os.path.exists(docs_path), 'Missing: {}'.format(docs_path)

    # Build inverted index chunks and reencode the documents
    tmp_dir = os.path.join(out_dir, 'index.tmp')
    index_path = os.path.join(out_dir, 'index.bin')
    data_dir = os.path.join(out_dir, 'data')
    remove_if_exists(index_path)
    remove_if_exists(data_dir)

    os.makedirs(data_dir)
    os.makedirs(tmp_dir)
    try:
        index_all_docs(docs_to_index, documents, lex_path, index_path, data_dir,
                       tmp_dir, chunk_size, parallelism, keep_tmp_files)
    finally:
        if not keep_tmp_files and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    assert os.path.exists(index_path), 'Missing: {}'.format(index_path)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
