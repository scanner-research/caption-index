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
from collections import defaultdict
from typing import List, Optional

from captions import Lexicon, Documents

from lib.common import (
    DocumentToIndex, read_docs_from_stdin, list_docs,
    get_word_counts, index_documents)

DEFAULT_OUT_DIR = 'out'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--doc-dir', type=str,
                   help='Directory containing captions. If not passed, read from stdin.')
    p.add_argument('-o', dest='out_dir', type=str, default=DEFAULT_OUT_DIR,
                   help='Output directory. Default: {}'.format(DEFAULT_OUT_DIR))
    p.add_argument('--chunk-size', dest='chunk_size', type=int,
                   help='Break the index into chunks of n documents')
    return p.parse_args()


def index_all_docs(
        docs_to_index: List[DocumentToIndex],
        documents: Documents,
        lexicon: Lexicon,
        index_out_path: str,
        data_out_dir: str,
        chunk_size: Optional[int],
):
    """Builds inverted indexes and reencode documents in binary"""
    assert len(docs_to_index) == len(documents)
    if chunk_size is not None:
        os.makedirs(index_out_path)

    index_and_doc_paths = defaultdict(list)
    for doc_to_index, doc in zip(docs_to_index, documents):
        assert doc_to_index.name == doc.name
        if chunk_size is None:
            doc_index_out_path = index_out_path
        elif chunk_size == 1:
            doc_index_out_path = os.path.join(
                index_out_path, '{:07d}.bin'.format(doc.id))
        else:
            base_idx = int(doc.id / chunk_size) * chunk_size
            doc_index_out_path = os.path.join(
                index_out_path, '{:07d}-{:07d}.bin'.format(
                    base_idx, min(base_idx + chunk_size, len(docs_to_index))))
        doc_data_out_path = os.path.join(
            data_out_dir, '{}.bin'.format(doc.id))

        index_and_doc_paths[doc_index_out_path].append(
            (doc.id, doc_to_index.path, doc_data_out_path))

    index_documents(list(index_and_doc_paths.items()), lexicon)


def build_lexicon(
        docs_to_index: List[DocumentToIndex],
        lex_path: str,
) -> Lexicon:
    print('Building lexicon: {}'.format(lex_path))
    word_counts = get_word_counts(docs_to_index)
    lexicon = Lexicon([
        Lexicon.Word(i, w, word_counts[w])
        for i, w in enumerate(sorted(word_counts.keys()))
    ])
    print('Storing lexicon: {}'.format(lex_path))
    lexicon.store(lex_path)
    return lexicon


def remove_if_exists(fpath):
    if os.path.isdir(fpath):
        shutil.rmtree(fpath)
    elif os.path.isfile(fpath):
        os.remove(fpath)


def main(
        out_dir: str, doc_dir: Optional[str],
        chunk_size: Optional[int] = None
):
    assert chunk_size is None or chunk_size > 0

    # Load document names
    if doc_dir:
        docs_to_index = list(sorted(list_docs(doc_dir)))
    else:
        docs_to_index = read_docs_from_stdin()

    os.makedirs(out_dir, exist_ok=True)

    # Load or build a lexicon
    lex_path = os.path.join(out_dir, 'lexicon.txt')
    if not os.path.exists(lex_path):
        lexicon = build_lexicon(docs_to_index, lex_path)
        assert os.path.exists(lex_path), 'Missing: {}'.format(lex_path)
    else:
        lexicon = Lexicon.load(lex_path)

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
    index_path = os.path.join(out_dir, 'index.bin')
    data_dir = os.path.join(out_dir, 'data')
    remove_if_exists(index_path)
    remove_if_exists(data_dir)

    os.makedirs(data_dir)
    index_all_docs(docs_to_index, documents, lexicon, index_path, data_dir,
                   chunk_size)

    assert os.path.exists(index_path), 'Missing: {}'.format(index_path)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
