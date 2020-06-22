#!/usr/bin/env python3

"""
Update an existing index

This will produce:
 - an updated document list
 - index file(s) for the additional documents
 - binary data file(s) for the additional documents
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

from lib.common import (
    DEFAULT_PARALLELISM, DocumentToIndex, read_docs_from_stdin, list_docs,
    merge_files, get_word_counts)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('index_dir', type=str,
                   help='Directory containing existing index')
    p.add_argument('-d', '--new-doc-dir', type=str,
                   help='Directory containing captions. If not passed, read from stdin.')
    p.add_argument('-j', dest='parallelism', type=int, default=DEFAULT_PARALLELISM,
                   help='Number of CPU cores to use. Default: {}'.format(DEFAULT_PARALLELISM))
    p.add_argument('--chunk-size', dest='chunk_size', type=int,
                   help='Break the index into chunks of n documents')
    p.add_argument('--skip-existing-names', action='store_true',
                   help='Skip documents that are already indexed')
    return p.parse_args()


def index_single_doc(args):
    doc_id, doc_path, index_out_path, data_out_path = args
    index_document(doc_id, doc_path, index_single_doc.lexicon, index_out_path,
                   data_out_path, tokenizer=AlignmentTokenizer())
index_single_doc.lexicon = None


def init_worker(function, lexicon_path):
    function.lexicon = Lexicon.load(lexicon_path)


def index_new_docs(
        new_docs_to_index: List[DocumentToIndex],
        new_documents: Documents,
        tmp_dir: str,
        lexicon_path: str,
        parallelism: int
):
    """Builds inverted indexes and reencode documents in binary"""
    assert len(new_docs_to_index) == len(new_documents)

    with Pool(
            processes=parallelism, initializer=init_worker,
            initargs=(index_single_doc, lexicon_path)
    ) as pool:
        index_out_paths = []
        data_out_paths = []
        worker_args = []
        for doc_to_index, doc in zip(new_docs_to_index, new_documents):
            assert doc_to_index.name == doc.name
            doc_index_out_path = os.path.join(
                tmp_dir, '{:07d}.ind'.format(doc.id))
            doc_data_out_path = os.path.join(
                tmp_dir, '{}.bin'.format(doc.id))
            worker_args.append((doc.id, doc_to_index.path, doc_index_out_path,
                                doc_data_out_path))
            index_out_paths.append(doc_index_out_path)
            data_out_paths.append(doc_data_out_path)

        for _ in tqdm(pool.imap_unordered(index_single_doc, worker_args),
                      desc='Indexing', total=len(worker_args)):
            pass

    return index_out_paths, data_out_paths


def main(
        index_dir: str,
        new_doc_dir: Optional[str],
        parallelism: int = DEFAULT_PARALLELISM,
        chunk_size: Optional[int] = None,
        skip_existing_names: bool = False
):
    assert chunk_size is None or chunk_size > 0
    assert parallelism > 0
    doc_path = os.path.join(index_dir, 'documents.txt')
    lex_path = os.path.join(index_dir, 'lexicon.txt')
    index_path = os.path.join(index_dir, 'index.bin')

    old_lexicon = Lexicon.load(lex_path)

    documents = Documents.load(doc_path)

    if new_doc_dir:
        new_docs_to_index = list_docs(new_doc_dir)
    else:
        new_docs_to_index = read_docs_from_stdin()

    assert len(new_docs_to_index) > 0
    tmp_new_docs_to_index = []
    for new_doc in new_docs_to_index:
        if new_doc.name in documents:
            if skip_existing_names:
                print('Skipping: {} is already indexed!'.format(new_doc.name))
            else:
                raise Exception(
                    '{} is already indexed! Aborting.'.format(new_doc.name))
        else:
            tmp_new_docs_to_index.append(new_doc)
    new_docs_to_index = tmp_new_docs_to_index
    if len(new_docs_to_index) == 0:
        print('No new documents to index.')
        return

    # Update lexicon
    new_word_counts = get_word_counts(new_docs_to_index, parallelism)
    lexicon_words = [
        Lexicon.Word(w.id, w.token, w.count + new_word_counts[w.token]
                     if w.token in new_word_counts else w.count)
        for w in old_lexicon
    ]
    for w in new_word_counts:
        if w not in old_lexicon:
            lexicon_words.append(
                Lexicon.Word(len(lexicon_words), w, new_word_counts[w]))

    tmp_lex_path = os.path.join(index_dir, 'lexicon.tmp')
    Lexicon(lexicon_words).store(tmp_lex_path)

    base_doc_id = len(documents)
    new_documents = [Documents.Document(id=i + base_doc_id, name=d.name)
                     for i, d in enumerate(new_docs_to_index)]

    # Index the new documents
    tmp_dir = os.path.join(index_dir, 'update-index.tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    new_doc_index_paths, new_doc_data_paths = index_new_docs(
        new_docs_to_index, new_documents, tmp_dir, tmp_lex_path, parallelism)

    # Convert existing index.bin to a dirctory if needed
    if os.path.isfile(index_path):
        tmp_index_path = index_path + '.tmp'
        shutil.move(index_path, tmp_index_path)
        os.makedirs(index_path)
        shutil.move(
            tmp_index_path,
            os.path.join(index_path, '{:07d}-{:07d}.bin'.format(
                0, base_doc_id)))

    assert os.path.isdir(index_path)
    if chunk_size is None:
        new_index_path = os.path.join(
            index_path, '{:07d}-{:07d}.bin'.format(
                base_doc_id, base_doc_id + len(new_doc_index_paths)))
        merge_files(new_doc_index_paths, new_index_path)
    else:
        max_doc_id = base_doc_id + len(new_doc_index_paths)
        for i in range(
                base_doc_id, max_doc_id, chunk_size
        ):
            new_index_path = os.path.join(
                index_path, '{:07d}-{:07d}.bin'.format(
                    i, min(i + chunk_size, max_doc_id)))
            merge_files(
                new_doc_index_paths[i:i + chunk_size], new_index_path)

    for data_path in new_doc_data_paths:
        shutil.move(data_path, os.path.join(index_dir, 'data'))

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Write out the new documents file
    shutil.move(doc_path, doc_path + '.old')
    all_documents = list(documents)
    all_documents.extend(new_documents)
    Documents(all_documents).store(doc_path)

    # Update to the new lexicon
    shutil.move(lex_path, lex_path + '.old')
    shutil.move(tmp_lex_path, lex_path)

    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
