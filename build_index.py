#!/usr/bin/env python3

import argparse
import heapq
import math
import pysrt
import os
import shutil
from collections import defaultdict, deque, namedtuple, Counter
from multiprocessing import Pool
from subprocess import check_call
from threading import Lock
from tqdm import tqdm

from util.index import tokenize, Lexicon, Documents, \
    encode_datum, decode_datum, encode_time_int, \
    DATUM_SIZE, TIME_INT_SIZE, MAX_TIME_INT_VALUE, MAX_DATUM_VALUE


DEFAULT_OUT_DIR = 'out'
DEFAULT_WORKERS = os.cpu_count()


TMP_INV_IDX_EXT = '.inv.bin'
TMP_BIN_DOC_EXT = '.doc.bin'


N_WORKERS = None


# Hack to get around sharing args beween processes workers
WORKER_LEXICON = None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('doc_dir', type=str, help='Directory containing transcripts')
    p.add_argument('-o', dest='out_dir', type=str, default=DEFAULT_OUT_DIR,
                   help='Output directory. Default: {}'.format(DEFAULT_OUT_DIR))
    p.add_argument('-j', dest='workers', type=int, default=DEFAULT_WORKERS,
                   help='Number of CPU cores to use. Default: {}'.format(DEFAULT_WORKERS))
    p.add_argument('--limit', dest='limit', type=int,
                   help='Number of documents to parse. Default: None')
    return p.parse_args()


def list_subs(dir):
    return [f for f in os.listdir(dir) if f.endswith('.srt')]


def load_srt(doc_path):
    try:
        subs = pysrt.open(doc_path)
    except:
        try:
            subs = pysrt.open(doc_path, encoding='iso-8859-1')
        except:
            raise Exception('Cannot parse {}'.format(doc_path))
    return subs


def get_doc_words(doc_path):
    try:
        subs = load_srt(doc_path)
    except Exception as e:
        print(e)
        return set()

    words = Counter()
    for s in subs:
        tokens = tokenize(s.text)
        words.update(tokens)
    return words


def get_words(doc_dir, doc_names):
    assert isinstance(doc_dir, str)
    assert isinstance(doc_names, list)

    words = Counter()
    words_lock = Lock()
    with tqdm(total=len(doc_names)) as pbar, Pool(processes=N_WORKERS) as pool:
        pbar.set_description('Building lexicon')

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


def index_single_doc(doc_path, lexicon):
    assert isinstance(lexicon, Lexicon)

    doc_inv_index = defaultdict(deque)  # token_id -> [postings]
    doc_lines = deque()                 # [(position, start, end, [tokens])]
    try:
        subs = load_srt(doc_path)
        doc_position = 0
        for s in subs:
            start, end = s.start.ordinal, s.end.ordinal
            if start > end:
                print('Warning: start time > end time ({} > {})'.format(start, end))
                end = start
            if end - start > MAX_TIME_INT_VALUE:
                print('Warning: end - start > {}ms'.format(MAX_TIME_INT_VALUE))
                end = start + MAX_TIME_INT_VALUE

            tokens = deque()
            entry_start_position = doc_position

            for t in tokenize(s.text):
                token = None
                try:
                    try:
                        token = lexicon[t]
                    except KeyError:
                        print('Unknown token: {}'.format(t))
                    doc_inv_index[token.id].append((doc_position, start, end))
                finally:
                    doc_position += 1
                    tokens.append(MAX_DATUM_VALUE if token is None else token.id)

            doc_lines.append((entry_start_position, start, end, tokens))
    except Exception as e:
        print(e)
    return doc_lines, doc_inv_index


def write_inv_index(inv_index, out_path):
    with open(out_path, 'wb') as f:
        # Order by increasing token_id
        for token_id in sorted(inv_index):
            docs = inv_index[token_id]
            assert len(docs) > 0

            f.write(encode_datum(token_id))    # Token id
            f.write(encode_datum(len(docs)))   # Num of docs in list

            # Order by increasing doc_id
            for doc_id, postings in sorted(docs, key=lambda x: x[0]):
                f.write(encode_datum(doc_id))          # Doc id
                f.write(encode_datum(len(postings)))   # Num postings in doc
                assert len(postings) > 0

                for (position, start, end) in postings:
                    f.write(encode_datum(position))
                    f.write(encode_time_int(start, end))


def write_doc_binary(batch_doc_lines, out_path):
    """Returns [(time_idx_offset, token_data_offset, doc_len)]"""
    batch_doc_offsets = deque()
    with open(out_path, 'wb') as f:
        for doc_id, doc_lines in batch_doc_lines:
            doc_time_idx_start = f.tell()
            for position, start, end, _ in doc_lines:
                f.write(encode_time_int(start, end))
                f.write(encode_datum(position))

            doc_len = 0
            doc_token_data_start = f.tell()
            for _, _, _, tokens in doc_lines:
                for t in tokens:
                    f.write(encode_datum(t))
                doc_len += len(tokens)
            batch_doc_offsets.append(
                (doc_time_idx_start, doc_token_data_start, doc_len))
    return batch_doc_offsets


def index_batch(doc_batch, out_path_prefix):
    """Build an inverted index and binary reencode documents"""
    lexicon = WORKER_LEXICON

    assert isinstance(out_path_prefix, str)
    assert isinstance(lexicon, Lexicon)

    batch_inv_index = defaultdict(deque)    # token_id -> [(doc, [postings])]
    batch_doc_lines = deque()
    for doc_id, doc_path in doc_batch:
        doc_lines, doc_inv_index = index_single_doc(doc_path, lexicon)

        for token_id, postings in doc_inv_index.items():
            batch_inv_index[token_id].append((doc_id, postings))

        batch_doc_lines.append((doc_id, doc_lines))

    write_inv_index(batch_inv_index, out_path_prefix + TMP_INV_IDX_EXT)
    batch_doc_offsets = write_doc_binary(batch_doc_lines,
                                         out_path_prefix + TMP_BIN_DOC_EXT)

    return batch_doc_offsets


def index_all_docs(doc_dir, doc_names, lexicon, bin_doc_path, out_dir,
                   batch_size=100):
    """Builds inverted indexes and reencode documents in binary"""

    assert isinstance(doc_names, list)
    assert isinstance(lexicon, Lexicon)

    global WORKER_LEXICON
    WORKER_LEXICON = lexicon

    with tqdm(total=len(doc_names)) as pbar, Pool(processes=N_WORKERS) as pool:
        async_results = deque()
        pbar.set_description('Building indexes')

        def progress(doc_offsets):
            pbar.update(len(doc_offsets))

        worker_args = deque()
        for base_id in range(0, len(doc_names), batch_size):
            doc_batch = [
                (i, os.path.join(doc_dir, doc_names[i]))
                for i in range(base_id, min(base_id + batch_size, len(doc_names)))
            ]
            out_path_prefix = os.path.join(out_dir, str(base_id))
            async_results.append(pool.apply_async(
                index_batch, (doc_batch, out_path_prefix), callback=progress))
            worker_args.append((doc_batch, out_path_prefix))

        # Merge the offsets and generate a new set of Documents with offsets
        # annotated
        documents = [None] * len(doc_names)
        global_offset = 0
        for args, async in zip(worker_args, async_results):
            doc_batch, out_path_prefix = args
            doc_offsets = async.get()
            for a, b in zip(doc_batch, doc_offsets):
                doc_id, _ = a
                doc_time_idx_offset, doc_token_data_offset, doc_length = b
                documents[doc_id] = Documents.Document(
                    id=doc_id, name=doc_names[doc_id], length=doc_length,
                    time_index_offset=doc_time_idx_offset + global_offset,
                    token_data_offset=doc_token_data_offset + global_offset)
            global_offset += os.path.getsize(out_path_prefix + TMP_BIN_DOC_EXT)

        assert None not in documents, 'Not all documents were indexed'

        # Cat the files together
        bin_doc_paths = [p + TMP_BIN_DOC_EXT for _, p in worker_args]
        with open(bin_doc_path, 'wb') as f:
            check_call(['cat'] + bin_doc_paths, stdout=f)

        return Documents(documents)


class MergeIndexParser(object):

    ParsedDocument = namedtuple('ParsedDocument', ['id', 'n', 'data'])

    def __init__(self, path, min_token, max_token):
        """Min and max are INCLUSIVE"""
        self._path = path
        self._min_token = min_token
        self._max_token = max_token
        self._f = open(path, 'rb')
        self._done = False
        self._curr_token = None
        self._curr_doc = None
        self._curr_n_docs = None
        self._curr_n_docs_left = 0
        self.next_token()

    @property
    def path(self):
        return self._path

    @property
    def token(self):
        return self._curr_token if not self._done else None

    @property
    def ndocs(self):
        assert not self._done, 'EOF already reached'
        return self._curr_n_docs

    @property
    def doc(self):
        assert not self._done, 'EOF already reached'
        return self._curr_doc

    def close(self):
        self._f.close()
        self._done = True
        self._curr_token = None
        self._curr_n_docs = None
        self._curr_n_docs_left = None

    def next_token(self):
        assert not self._done, 'EOF already reached'
        assert self._curr_n_docs_left == 0, 'Not done processing docs'
        while True:
            token_data = self._f.read(DATUM_SIZE)
            if len(token_data) == 0:
                self.close()
                break
            next_token = decode_datum(token_data)
            if next_token > self._max_token:
                self.close()
                break

            self._curr_token = next_token
            self._curr_n_docs = decode_datum(self._f.read(DATUM_SIZE))
            self._curr_n_docs_left = self._curr_n_docs
            assert self._curr_n_docs > 0, 'Token cannot have no docs'
            if self._curr_token < self._min_token:
                self.skip_docs()
            else:
                assert self.next_doc() is not None
                break
        return self.token

    def next_doc(self):
        assert self._curr_n_docs_left >= 0
        if self._curr_n_docs_left == 0:
            self._curr_doc = None
        else:
            self._curr_n_docs_left -= 1
            doc_id = decode_datum(self._f.read(DATUM_SIZE))
            n_postings = decode_datum(self._f.read(DATUM_SIZE))
            assert n_postings > 0, 'Empty postings list'
            postings_len = n_postings * (DATUM_SIZE + TIME_INT_SIZE)
            postings_data = self._f.read(postings_len)
            assert len(postings_data) == postings_len, 'Invalid read'
            self._curr_doc = MergeIndexParser.ParsedDocument(
                id=doc_id, n=n_postings, data=postings_data)
        return self.doc

    def skip_docs(self):
        assert self._curr_n_docs_left >= 0
        while self._curr_n_docs_left > 0:
            self._f.seek(DATUM_SIZE, 1)
            n_postings = decode_datum(self._f.read(DATUM_SIZE))
            postings_len = n_postings * (DATUM_SIZE + TIME_INT_SIZE)
            assert n_postings > 0, 'Empty postings list'
            self._f.seek(postings_len, 1)
            self._curr_n_docs_left -= 1

    def __lt__(self, o):
        # For priority queuing
        if self.token == o.token:
            return self.doc < o.doc
        return self.token < o.token


def merge_inv_indexes(idx_paths, out_path, min_token, max_token):
    lexicon = WORKER_LEXICON

    token_parsers_pq = []
    for path in idx_paths:
        p = MergeIndexParser(path, min_token, max_token - 1)
        if p.token is not None:
            heapq.heappush(token_parsers_pq, p)

    jump_offsets = [-1] * len(lexicon)
    with open(out_path, 'wb') as f:
        for _ in range(len(lexicon)):
            if len(token_parsers_pq) == 0:
                break
            token_id = token_parsers_pq[0].token
            jump_offsets[token_id] = f.tell()

            doc_parsers_pq = []
            doc_count = 0
            while True:
                if (len(token_parsers_pq) == 0 or
                        token_parsers_pq[0].token != token_id):
                    break
                p = heapq.heappop(token_parsers_pq)
                doc_count += p.ndocs
                heapq.heappush(doc_parsers_pq, p)
            assert len(doc_parsers_pq) >= 1

            f.write(encode_datum(token_id))   # Token id
            f.write(encode_datum(doc_count))  # Num docs

            while len(doc_parsers_pq) > 0:
                p = heapq.heappop(doc_parsers_pq)
                f.write(encode_datum(p.doc.id))      # Doc id
                f.write(encode_datum(p.doc.n))       # Num postings
                f.write(p.doc.data)                  # Raw postings data

                if p.next_doc() is None:
                    if p.next_token() is not None:
                        # Return to tokens queue
                        heapq.heappush(token_parsers_pq, p)
                else:
                    # Return to docs queue
                    heapq.heappush(doc_parsers_pq, p)

    assert len(token_parsers_pq) == 0, 'Uh oh... still have lexicons to merge'
    return jump_offsets


def parallel_merge_inv_indexes(idx_dir, lexicon, out_path, merge_dir):

    assert isinstance(lexicon, Lexicon)
    global WORKER_LEXICON
    WORKER_LEXICON = lexicon

    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)

    try:
        idx_paths = [
            os.path.join(idx_dir, x)
            for x in os.listdir(idx_dir) if x.endswith(TMP_INV_IDX_EXT)
        ]

        with tqdm(total=N_WORKERS) as pbar:
            pbar.set_description('Merging indexes')

            def progress(ignored):
                pbar.update(1)

            with Pool(processes=N_WORKERS + 1) as pool:
                tokens_per_worker = math.ceil(len(lexicon) / N_WORKERS)
                new_idx_paths = []
                async_results = deque()

                # Partition across the lexicon
                for file_no, i in enumerate(range(0, len(lexicon),
                                            tokens_per_worker)):
                    new_idx_path = os.path.join(
                        merge_dir, '{}{}'.format(file_no, TMP_INV_IDX_EXT))
                    new_idx_paths.append(new_idx_path)
                    async_results.append(pool.apply_async(
                        merge_inv_indexes,
                        (
                            idx_paths, new_idx_path,
                            i, min(i + tokens_per_worker, len(lexicon))
                        ),
                        callback=progress
                    ))

                # Merge the jump offsets
                jump_offsets = [-1] * len(lexicon)
                global_offset = 0
                for i, (new_idx_path, async) in enumerate(
                        zip(new_idx_paths, async_results)):
                    worker_jump_offsets = async.get()
                    min_token = i * tokens_per_worker
                    max_token = min(min_token + tokens_per_worker, len(lexicon))
                    for t in range(min_token, max_token):
                        jump_offsets[t] = global_offset + worker_jump_offsets[t]
                    global_offset += os.path.getsize(new_idx_path)

                # Cat all the files together
                with open(out_path, 'wb') as f:
                    check_call(['cat'] + new_idx_paths, stdout=f)
                progress(None)

            if -1 in jump_offsets:
                print('Warning: not all lexicon words have been indexed')
            return jump_offsets
    finally:
        shutil.rmtree(merge_dir, True)


def main(doc_dir, out_dir, workers, limit=None):
    global N_WORKERS
    N_WORKERS = workers

    # Load document names
    doc_names = list(sorted(list_subs(doc_dir)))
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
        # Save lexicon without offsets initally
        wordcounts = get_words(doc_dir, doc_names)
        lexicon = Lexicon([
            Lexicon.Word(i, w, wordcounts[w], -1)
            for i, w in enumerate(sorted(wordcounts.keys()))
        ])
        lexicon.store(lex_path)
        del wordcounts

    # Build inverted index chunks and reencode the documents
    chunk_dir = os.path.join(out_dir, 'chunks')
    docs_path = os.path.join(out_dir, 'docs.list')
    bin_docs_path = os.path.join(out_dir, 'docs.bin')
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
        try:
            documents = index_all_docs(doc_dir, doc_names, lexicon,
                                       bin_docs_path, chunk_dir)

            # Store the document list
            print('Storing document list: {}'.format(docs_path))
            documents.store(docs_path)
            del documents
        except:
            shutil.rmtree(chunk_dir)
            raise
    else:
        print('Found existing indexes: {}'.format(chunk_dir))
    assert os.path.exists(docs_path), 'Missing: {}'.format(docs_path)
    assert os.path.exists(bin_docs_path), 'Missing: {}'.format(bin_docs_path)

    # Merge the inverted index chunks
    idx_path = os.path.join(out_dir, 'index.bin')
    merge_dir = os.path.join(out_dir, 'merge')
    jump_offsets = parallel_merge_inv_indexes(chunk_dir, lexicon, idx_path,
                                              merge_dir)
    assert os.path.exists(idx_path), 'Missing: {}'.format(idx_path)

    # Resave the lexicon with offsets
    print('Storing lexicon with offsets: {}'.format(lex_path))
    lexicon = Lexicon([
        Lexicon.Word(w.id, w.token, w.count, jump_offsets[w.id])
        for w in lexicon
    ])
    lexicon.store(lex_path)
    assert os.path.exists(lex_path), 'Missing: {}'.format(idx_path)


if __name__ == '__main__':
    main(**vars(get_args()))
