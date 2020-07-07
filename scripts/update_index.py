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
from collections import defaultdict
import shutil
from typing import List, Optional

from captions import Lexicon, Documents

from lib.common import (
    DocumentToIndex, read_docs_from_stdin, list_docs,
    merge_files, get_word_counts, index_documents)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('index_dir', type=str,
                   help='Directory containing existing index')
    p.add_argument('-d', '--new-doc-dir', type=str,
                   help='Directory containing captions. If not passed, read from stdin.')
    p.add_argument('--chunk-size', dest='chunk_size', type=int,
                   help='Break the index into chunks of n documents')
    p.add_argument('--skip-existing-names', action='store_true',
                   help='Skip documents that are already indexed')
    return p.parse_args()


def index_new_docs(
        new_docs_to_index: List[DocumentToIndex],
        new_documents: Documents,
        lexicon: Lexicon,
        index_dir: str,
        data_dir: str,
        chunk_size: Optional[int]
):
    """Builds inverted indexes and reencode documents in binary"""
    assert len(new_docs_to_index) == len(new_documents)
    base_doc_id = min(d.id for d in new_documents)
    max_doc_id = max(d.id for d in new_documents)
    index_and_doc_paths = defaultdict(list)
    for doc_to_index, doc in zip(new_docs_to_index, new_documents):
        assert doc_to_index.name == doc.name

        if chunk_size is None:
            doc_index_out_path = os.path.join(
                index_dir, '{:07d}-{:07d}.bin'.format(
                    base_doc_id, base_doc_id + len(new_docs_to_index)))
        elif chunk_size == 1:
            doc_index_out_path = os.path.join(
                index_dir, '{:07d}.bin'.format(doc.id))
        else:
            chunk_idx = int((doc.id - base_doc_id) / chunk_size) * chunk_size
            doc_index_out_path = os.path.join(
                index_dir, '{:07d}-{:07d}.bin'.format(
                    base_doc_id + chunk_idx,
                    min(base_doc_id + chunk_idx + chunk_size, max_doc_id)))

        doc_data_out_path = os.path.join(
            data_dir, '{}.bin'.format(doc.id))
        index_and_doc_paths[doc_index_out_path].append(
            (doc.id, doc_to_index.path, doc_data_out_path))

    index_documents(list(index_and_doc_paths.items()), lexicon)


def main(
        index_dir: str,
        new_doc_dir: Optional[str],
        chunk_size: Optional[int] = None,
        skip_existing_names: bool = False
):
    assert chunk_size is None or chunk_size > 0
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
    new_word_counts = get_word_counts(new_docs_to_index)
    lexicon_words = [
        Lexicon.Word(w.id, w.token, w.count + new_word_counts[w.token]
                     if w.token in new_word_counts else w.count)
        for w in old_lexicon
    ]
    for w in new_word_counts:
        if w not in old_lexicon:
            lexicon_words.append(
                Lexicon.Word(len(lexicon_words), w, new_word_counts[w]))
    lexicon = Lexicon(lexicon_words)

    base_doc_id = len(documents)
    new_documents = [Documents.Document(id=i + base_doc_id, name=d.name)
                     for i, d in enumerate(new_docs_to_index)]

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

    # Index the new documents
    index_new_docs(new_docs_to_index, new_documents, lexicon, index_path,
                   os.path.join(index_dir, 'data'), chunk_size)

    # Write out the new documents file
    shutil.move(doc_path, doc_path + '.old')
    all_documents = list(documents)
    all_documents.extend(new_documents)
    Documents(all_documents).store(doc_path)

    # Update to the new lexicon
    lexicon.store(lex_path)

    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
