#!/usr/bin/env python3

import os
import sys
import shutil
import tempfile
import unittest
from subprocess import check_call

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../scripts')

from index import BinaryFormat, Lexicon, Documents, InvertedIndex, DocumentData
from build import main as build_main
from scan import main as scan_main
from search import main as search_main


TMP_DIR = None
TEST_SUBS_SUBDIR = 'subs'
TEST_INDEX_SUBDIR = 'index'
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test.tar.gz')


def build_test_index(tmp_dir):
    subs_dir = os.path.join(tmp_dir, TEST_SUBS_SUBDIR)
    idx_dir = os.path.join(tmp_dir, TEST_INDEX_SUBDIR)

    # Unpack the test data
    os.makedirs(subs_dir)
    check_call(['tar', '-xzf', TEST_DATA_PATH, '-C', subs_dir])

    build_main(subs_dir, idx_dir, 1)
    assert os.path.isdir(idx_dir)


def get_docs_and_lex(idx_dir):
    doc_path = os.path.join(idx_dir, 'docs.list')
    lex_path = os.path.join(idx_dir, 'words.lex')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)
    return documents, lexicon


class TestBinaryFormat(unittest.TestCase):

    def test_datum(self):
        bf = BinaryFormat.default()
        self.assertEqual(0, bf.decode_datum(bf.encode_datum(0)))
        self.assertEqual(111, bf.decode_datum(bf.encode_datum(111)))
        self.assertEqual(
            bf.max_datum_value,
            bf.decode_datum(bf.encode_datum(bf.max_datum_value)))

    def test_time_interval(self):
        bf = BinaryFormat.default()
        self.assertEqual((0, 0), bf.decode_time_interval(
                         bf.encode_time_interval(0, 0)))
        self.assertEqual((0, 100), bf.decode_time_interval(
                         bf.encode_time_interval(0, 100)))
        self.assertEqual((777, 888), bf.decode_time_interval(
                         bf.encode_time_interval(777, 888)))
        self.assertEqual((76543210, 76543210 + bf.max_time_interval),
                         bf.decode_time_interval(bf.encode_time_interval(
                            76543210, 76543210 + bf.max_time_interval)))

    def test_byte(self):
        bf = BinaryFormat.default()
        for i in range(256):
            self.assertEqual(i, int.from_bytes(bf.encode_byte(i), bf._endian))


class TestInvertedIndex(unittest.TestCase):

    def test_index(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        idx_path = os.path.join(idx_dir, 'index.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with InvertedIndex(idx_path, lexicon, documents) as index:
            # Unigram search
            r = index.search('THE')
            for i, d in enumerate(r.documents):
                self.assertEqual(d.count, len(list(d.locations)))
            self.assertEqual(i + 1, r.count)

            # N-gram search
            r = index.search('UNITED STATES')
            for d in r.documents:
                for l in d.locations:
                    pass

            # N-gram search
            r = index.search('UNITED STATES OF AMERICA')
            for d in r.documents:
                for l in d.locations:
                    pass


class TestDocumentData(unittest.TestCase):

    def test_token_data(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        data_path = os.path.join(idx_dir, 'docs.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with DocumentData(data_path, lexicon, documents) as docdata:
            for i in range(len(documents)):
                for t in docdata.tokens(i):
                    pass
                for t in docdata.tokens(i, decode=True):
                    pass

    def test_time_index(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        data_path = os.path.join(idx_dir, 'docs.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with DocumentData(data_path, lexicon, documents) as docdata:
            for i in range(len(documents)):
                for interval in docdata.token_intervals(i, 0, 2 ** 16):
                    for t in interval.tokens:
                        pass
                for interval in docdata.token_intervals(i, 0, 0):
                    for t in interval.tokens:
                        pass
                for interval in docdata.token_intervals(i, 0, 2 ** 16,
                                                        decode=True):
                    for t in interval.tokens:
                        pass
                for interval in docdata.token_intervals(i, 0, 0, decode=True):
                    for t in interval.tokens:
                        pass


class TestScripts(unittest.TestCase):

    def test_scan(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        scan_main(idx_dir, os.cpu_count(), None)

    def test_search(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        search_main(idx_dir, ['UNITED', 'STATES'], False, 3)


if __name__ == '__main__':
    TMP_DIR = tempfile.mkdtemp(suffix=None, prefix='caption-index-unittest-',
                               dir=None)
    try:
        build_test_index(TMP_DIR)
        unittest.main()
    finally:
        shutil.rmtree(TMP_DIR, True)
