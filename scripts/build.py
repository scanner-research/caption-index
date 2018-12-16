#!/usr/bin/env python3

"""
Index a directory of transcript files
"""

import argparse
import pysrt
import os
import shutil
import traceback
from collections import defaultdict, deque, Counter
from io import BytesIO
from multiprocessing import Pool
from subprocess import check_call
from threading import Lock
from tqdm import tqdm
from typing import Dict, List, Tuple

from captions import tokenize, Lexicon, Documents, BinaryFormat

BINARY_FORMAT = BinaryFormat.default()
MAX_WORD_LEN = 20


DEFAULT_OUT_DIR = 'out'
DEFAULT_WORKERS = os.cpu_count()


N_WORKERS = None


# Hack to get around sharing args beween processes workers
WORKER_LEXICON = None


DEFAULT_SOURCE_FILE_EXT = 'srt'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('doc_dir', type=str,
                   help='Directory containing transcripts')
    p.add_argument('-o', dest='out_dir', type=str, default=DEFAULT_OUT_DIR,
                   help='Output directory. Default: {}'.format(
                        DEFAULT_OUT_DIR))
    p.add_argument('-j', dest='workers', type=int, default=DEFAULT_WORKERS,
                   help='Number of CPU cores to use. Default: {}'.format(
                        DEFAULT_WORKERS))
    p.add_argument('--ext', dest='extension',
                   default=DEFAULT_SOURCE_FILE_EXT,
                   help='Subtitle file extension. Default: {}'.format(
                        DEFAULT_SOURCE_FILE_EXT))
    p.add_argument('--limit', dest='limit', type=int,
                   help='Number of documents to parse. Default: None')
    return p.parse_args()


def list_subs(dir: str, ext: str):
    return [f for f in os.listdir(dir) if f.endswith(ext)]


def load_srt(doc_path: str):
    try:
        subs = pysrt.open(doc_path)
    except:
        try:
            subs = pysrt.open(doc_path, encoding='iso-8859-1')
        except:
            raise Exception('Cannot parse {}'.format(doc_path))
    return subs


def get_doc_words(doc_path: str):
    words = Counter()
    try:
        subs = load_srt(doc_path)
    except Exception as e:
        print(e)
        return words

    for s in subs:
        tokens = tokenize(s.text)
        words.update(t for t in tokens if len(t) <= MAX_WORD_LEN)
    return words


def get_words(doc_dir: str, doc_names: List[str]):
    words = Counter()
    words_lock = Lock()
    with tqdm(total=len(doc_names), desc='Building lexicon') as pbar, \
            Pool(processes=N_WORKERS) as pool:

        def collect(result):
            with words_lock:
                words.update(result)
            pbar.update(1)

        async_results = deque()
        for d in doc_names:
            doc_path = os.path.join(doc_dir, d)
            async_results.append(
                pool.apply_async(get_doc_words, (doc_path,), callback=collect))

        # Forces exceptions to be rethrown
        for a in async_results:
            a.get()

    print('Lexicon size: {}'.format(len(words)))
    return words


def millis_to_seconds(t: int):
    return t / 1000


def read_single_doc(doc_path: str, lexicon: Lexicon):
    doc_inv_index = defaultdict(deque)  # token_id -> [postings]
    doc_lines = deque()                 # [(position, start, end, [tokens])]
    try:
        subs = load_srt(doc_path)
        doc_position = 0
        for s in subs:
            start, end = s.start.ordinal, s.end.ordinal
            if start > end:
                print('Warning: start time > end time ({} > {})'.format(
                      start, end))
                end = start
            if end - start > BINARY_FORMAT.max_time_interval:
                print('Warning: end - start > {}ms'.format(
                      BINARY_FORMAT.max_time_interval))
                end = start + BINARY_FORMAT.max_time_interval

            tokens = deque()
            entry_start_position = doc_position

            for t in tokenize(s.text):
                token = None
                try:
                    try:
                        token = lexicon[t]
                    except KeyError:
                        print('Unknown token: {}'.format(t))
                        continue
                    doc_inv_index[token.id].append((doc_position, start, end))
                finally:
                    doc_position += 1
                    tokens.append(
                        BINARY_FORMAT.max_datum_value
                        if token is None else token.id)

            doc_lines.append((entry_start_position, start, end, tokens))

        if len(doc_lines) == 0:
            print('Empty file: {}'.format(doc_path))
    except Exception as e:
        print('Failed to index: {}'.format(doc_path))
        traceback.print_exc()
    return doc_inv_index, doc_lines


def write_doc_index(doc_id: int, doc_inv_index: Dict[int, Tuple[int, int, int]],
                    doc_lines: List[Tuple[int, int, int, List[int]]],
                    out_path: str):
    f_tokens = BytesIO()
    f_inv_index = BytesIO()

    doc_unique_token_count = len(doc_inv_index)
    doc_line_count = len(doc_lines)

    doc_posting_count = 0
    for token_id in sorted(doc_inv_index):
        f_tokens.write(BINARY_FORMAT.encode_datum(token_id))
        f_tokens.write(BINARY_FORMAT.encode_datum(doc_posting_count))

        postings = doc_inv_index[token_id]
        assert len(postings) > 0

        for (position, start, end) in postings:
            f_inv_index.write(BINARY_FORMAT.encode_time_interval(start, end))
            f_inv_index.write(BINARY_FORMAT.encode_datum(position))
            doc_posting_count += 1

    f_time_index = BytesIO()
    for position, start, end, _ in doc_lines:
        f_time_index.write(BINARY_FORMAT.encode_time_interval(start, end))
        f_time_index.write(BINARY_FORMAT.encode_datum(position))

    doc_len = 0
    doc_duration = 0
    f_data = BytesIO()
    for _, _, end, tokens in doc_lines:
        for t in tokens:
            f_data.write(BINARY_FORMAT.encode_datum(t))
        doc_len += len(tokens)
        doc_duration = max(doc_duration, end)

    # Checks to make sure that the lengths are correct
    assert doc_unique_token_count == f_tokens.tell() / (
        2 * BINARY_FORMAT.datum_bytes)
    assert doc_posting_count == f_inv_index.tell() / (
        BINARY_FORMAT.datum_bytes + BINARY_FORMAT.time_interval_bytes)
    assert doc_line_count == f_time_index.tell() / (
        BINARY_FORMAT.datum_bytes + BINARY_FORMAT.time_interval_bytes)
    assert doc_len == f_data.tell() / BINARY_FORMAT.datum_bytes

    # Write the index for the single document
    with open(out_path, 'wb') as f:
        f.write(BINARY_FORMAT.encode_u32(doc_id))
        f.write(BINARY_FORMAT.encode_u32(doc_duration))
        f.write(BINARY_FORMAT.encode_u32(doc_unique_token_count))
        f.write(BINARY_FORMAT.encode_u32(doc_posting_count))
        f.write(BINARY_FORMAT.encode_u32(doc_line_count))
        f.write(BINARY_FORMAT.encode_u32(doc_len))
        f.write(f_tokens.getvalue())
        f.write(f_inv_index.getvalue())
        f.write(f_time_index.getvalue())
        f.write(f_data.getvalue())


def index_single_doc(doc_id: int, doc_path: str, out_path: str):
    doc_inv_index, doc_lines = read_single_doc(doc_path, WORKER_LEXICON)
    write_doc_index(doc_id, doc_inv_index, doc_lines, out_path)


def index_all_docs(doc_dir: str, documents: Documents, lexicon: Lexicon,
                   out_file: str, tmp_dir: str):
    """Builds inverted indexes and reencode documents in binary"""

    global WORKER_LEXICON
    WORKER_LEXICON = lexicon

    with tqdm(total=len(documents), desc='Building indexes') as pbar, \
            Pool(processes=N_WORKERS) as pool:

        def progress(ignored):
            pbar.update(1)

        results = deque()
        for doc in documents:
            doc_path = os.path.join(doc_dir, doc.name)
            out_path = os.path.join(tmp_dir, str(doc.id))
            async = pool.apply_async(
                index_single_doc, (doc.id, doc_path, out_path),
                callback=progress)
            results.append((async, out_path))

        for async, _ in results:
            async.get()

        # Cat the files together
        doc_index_paths = [x for _, x in results]
        with open(out_file, 'wb') as f:
            check_call(['cat'] + doc_index_paths, stdout=f)


def main(doc_dir, out_dir, workers, extension=DEFAULT_SOURCE_FILE_EXT,
         limit=None):
    global N_WORKERS
    N_WORKERS = workers

    # Load document names
    doc_names = list(sorted(list_subs(doc_dir, extension)))
    if limit is not None:
        doc_names = doc_names[:limit]

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Load or build a lexicon
    lex_path = os.path.join(out_dir, 'words.lex')
    if os.path.exists(lex_path):
        print('Loading lexicon: {}'.format(lex_path))
        lexicon = Lexicon.load(lex_path)
    else:
        wordcounts = get_words(doc_dir, doc_names)
        lexicon = Lexicon([
            Lexicon.Word(i, w, wordcounts[w])
            for i, w in enumerate(sorted(wordcounts.keys()))
        ])
        lexicon.store(lex_path)
        del wordcounts

    # Build and store the document list
    docs_path = os.path.join(out_dir, 'docs.list')
    documents = Documents([
        Documents.Document(id=i, name=d) for i, d in enumerate(doc_names)
    ])
    print('Storing document list: {}'.format(docs_path))
    documents.store(docs_path)
    assert os.path.exists(docs_path), 'Missing: {}'.format(docs_path)

    # Build inverted index chunks and reencode the documents
    tmp_dir = os.path.join(out_dir, 'tmp')
    index_path = os.path.join(out_dir, 'index.bin')
    os.makedirs(tmp_dir)
    try:
        index_all_docs(doc_dir, documents, lexicon, index_path, tmp_dir)
    except:
        shutil.rmtree(tmp_dir)
        raise
    assert os.path.exists(index_path), 'Missing: {}'.format(index_path)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
