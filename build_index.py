#!/usr/bin/env python3

import argparse
import os
import pysrt
import heapq
from tqdm import tqdm
from threading import Lock
from multiprocessing import Pool
from collections import defaultdict, namedtuple, Counter

from util.index import tokenize, Lexicon, Documents


N_WORKERS = None

ENDIAN = 'little'
DATUM_SIZE = 4
MAX_INT = 2 ** (DATUM_SIZE * 8) - 1


def encode_datum(i):
    assert isinstance(i, int)
    assert i >= 0 and i <= MAX_INT, 'Out of range: {}'.format(i)
    return (i).to_bytes(DATUM_SIZE, ENDIAN)


def decode_datum(s):
    assert len(s) == DATUM_SIZE, '{} is too short'.format(len(s))
    return int.from_bytes(s, ENDIAN)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('doc_dir', type=str)
    p.add_argument('-o', dest='out_dir', type=str, default='build')
    p.add_argument('-j', dest='workers', type=int, default=os.cpu_count())
    p.add_argument('--limit', dest='limit', type=int)
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


def get_words(doc_dir, docs):
    assert isinstance(doc_dir, str)
    assert isinstance(docs, Documents)

    words = Counter()
    words_lock = Lock()
    with tqdm(total=len(docs)) as pbar, Pool(processes=N_WORKERS) as pool:
        pbar.set_description('Building lexicon')

        def collect(result):
            with words_lock:
                words.update(result)
            pbar.update(1)

        for d in docs:
            doc_path = os.path.join(doc_dir, d)
            if N_WORKERS > 1:
                pool.apply_async(get_doc_words, (doc_path,), callback=collect)
            else:
                collect(get_doc_words(doc_path))

        pool.close()
        pool.join()

    print('Lexicon size: {}'.format(len(words)))
    return words


def reverse_index_single_doc(doc_path, lexicon):
    assert isinstance(lexicon, Lexicon)

    doc_rindex = defaultdict(list)
    try:
        subs = load_srt(doc_path)
        doc_position = 0
        for s in subs:
            start, end = s.start.ordinal, s.end.ordinal
            for t in tokenize(s.text):
                token = lexicon[t]
                if token is None:
                    print('Unknown token: {}'.format(t))
                else:
                    doc_rindex[token.id].append((doc_position, start, end))
                doc_position += 1
    except Exception as e:
        print(e)
    return doc_rindex


# Hack to get around sharing args beween processes workers
WORKER_LEXICON = None


def reverse_index_batch(doc_batch, out_path):
    lexicon = WORKER_LEXICON

    assert isinstance(out_path, str)
    assert isinstance(lexicon, Lexicon)

    batch_rindex = defaultdict(list)     # token -> [(doc, [postings])]
    for doc_id, doc_path in doc_batch:
        doc_index = reverse_index_single_doc(doc_path, lexicon)
        for token_id, postings in doc_index.items():
            batch_rindex[token_id].append((doc_id, postings))

    with open(out_path, 'wb') as f:
        # Order by increasing token_id
        for token_id in sorted(batch_rindex):
            docs = batch_rindex[token_id]
            assert len(docs) > 0

            f.write(encode_datum(token_id))    # Token id
            f.write(encode_datum(len(docs)))   # Num of docs in list

            # Order by increasing doc_id
            for doc_id, postings in sorted(docs, key=lambda x: x[0]):
                f.write(encode_datum(doc_id))          # Doc id
                f.write(encode_datum(len(postings)))   # Num postings in doc
                assert len(postings) > 0

                for (position, start, end) in postings:
                    f.write(encode_datum(position))    # Position in document
                    f.write(encode_datum(start))       # Start time in ms
                    f.write(encode_datum(end))         # End time in ms

    return len(doc_batch)


def reverse_index_all_docs(doc_dir, docs, lexicon, out_dir, batch_size=100):
    assert isinstance(docs, Documents)
    assert isinstance(lexicon, Lexicon)

    global WORKER_LEXICON
    WORKER_LEXICON = lexicon

    with tqdm(total=len(docs)) as pbar, Pool(processes=N_WORKERS) as pool:
        async_results = []
        pbar.set_description('Building inverted index')

        def progress(n_indexed):
            pbar.update(n_indexed)

        for base_id in range(0, len(docs), batch_size):
            doc_batch = [
                (i, os.path.join(doc_dir, docs[i]))
                for i in range(base_id, min(base_id + batch_size, len(docs)))
            ]
            out_path = os.path.join(out_dir, '{}.bin'.format(base_id))

            if N_WORKERS > 1:
                async_results.append(pool.apply_async(
                    reverse_index_batch,
                    (doc_batch, out_path),
                    callback=progress))
            else:
                progress(reverse_index_batch(doc_batch, out_path))

        for async in async_results:
            async.wait()
            assert async.successful()

        pool.close()
        pool.join()


class MergeIndexParser(object):

    Document = namedtuple('Document', ['id', 'n', 'data'])

    def __init__(self, path):
        self._path = path
        self._f = open(path, 'rb')
        self._eof = False
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
        return self._curr_token if not self._eof else None

    @property
    def ndocs(self):
        assert not self._eof, 'EOF already read'
        return self._curr_n_docs

    @property
    def doc(self):
        assert not self._eof, 'EOF already read'
        return self._curr_doc

    def next_token(self):
        assert not self._eof, 'EOF already read'
        assert self._curr_n_docs_left == 0, 'Not done processing docs'
        token_data = self._f.read(DATUM_SIZE)
        if len(token_data) == 0:
            self._f.close()
            self._eof = True
            self._curr_token = None
            self._curr_n_docs = None
            self._curr_n_docs_left = None
        else:
            ndocs_data = self._f.read(DATUM_SIZE)
            self._curr_token = decode_datum(token_data)
            self._curr_n_docs = decode_datum(ndocs_data)
            self._curr_n_docs_left = self._curr_n_docs
            assert self._curr_n_docs > 0, 'Token cannot have no docs'
            assert self.next_doc() is not None
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
            postings_len = n_postings * DATUM_SIZE * 3  # (position, start, end)
            postings_data = self._f.read(postings_len)
            assert len(postings_data) == postings_len, 'Invalid read'
            self._curr_doc = MergeIndexParser.Document(
                id=doc_id, n=n_postings, data=postings_data)
        return self.doc

    def __lt__(self, o):
        # For priority queuing
        if self.token == o.token:
            return self.doc < o.doc
        return self.token == o.token


def merge_reverse_indexes(ridx_dir, lexicon, out_path):
    assert isinstance(lexicon, Lexicon)

    ridx_paths = [
        os.path.join(ridx_dir, x)
        for x in os.listdir(ridx_dir) if x.endswith('.bin')
    ]
    token_parsers_pq = []
    for path in ridx_paths:
        heapq.heappush(token_parsers_pq, MergeIndexParser(path))

    jump_offsets = [-1] * len(lexicon)
    with open(out_path, 'wb') as f, tqdm(total=len(lexicon)) as pbar:
        for i in range(len(lexicon)):
            if len(token_parsers_pq) == 0:
                break
            token_id = token_parsers_pq[0].token
            pbar.set_description(
                'Merging indexes ("{}")'.format(lexicon[token_id].token))
            pbar.update(1)
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

    return jump_offsets


def main(doc_dir, out_dir, workers, limit=None):
    global N_WORKERS
    N_WORKERS = workers

    docs = list(sorted(list_subs(doc_dir)))
    if limit is not None:
        docs = docs[:limit]
    docs = Documents(docs)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    lex_path = os.path.join(out_dir, 'words.lex')
    if os.path.exists(lex_path):
        print('Loading lexicon: {}'.format(lex_path))
        lexicon = Lexicon.load(lex_path)
    else:
        # Save lexicon without offsets initally
        wordcounts = get_words(doc_dir, docs)
        lexicon = Lexicon([
            Lexicon.Word(i, w, wordcounts[w], -1)
            for i, w in enumerate(sorted(wordcounts.keys()))
        ])
        lexicon.store(lex_path)
        del wordcounts

    doc_list_path = os.path.join(out_dir, 'documents.txt')
    docs.store(doc_list_path)

    chunk_dir = os.path.join(out_dir, 'parts')
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
        reverse_index_all_docs(doc_dir, docs, lexicon, chunk_dir)
    else:
        print('Exists: {}'.format(chunk_dir))

    idx_path = os.path.join(out_dir, 'index.bin')
    jump_offsets = merge_reverse_indexes(chunk_dir, lexicon, idx_path)

    # Resave the lexicon with offsets
    lexicon = Lexicon([
        Lexicon.Word(w.id, w.token, w.count, jump_offsets[w.id])
        for w in lexicon
    ])
    lexicon.store(lex_path)


if __name__ == '__main__':
    main(**vars(get_args()))
