#!/usr/bin/env python3

import os
import sys
import shutil
import tempfile
import unittest
from subprocess import check_call

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../scripts')

import build
import scan
import search
import build_metadata
import build_metadata
import index
import utils


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

    build.main(subs_dir, idx_dir, 1)
    assert os.path.isdir(idx_dir)


def get_docs_and_lex(idx_dir):
    doc_path = os.path.join(idx_dir, 'docs.list')
    lex_path = os.path.join(idx_dir, 'words.lex')

    documents = index.Documents.load(doc_path)
    lexicon = index.Lexicon.load(lex_path)
    return documents, lexicon


class TestTokenize(unittest.TestCase):

    def test_tokenize(self):
        text = 'I\'m a string! This is is a tokenizer test.'
        tokens = list(index.tokenize(text))
        self.assertTrue(isinstance(tokens[0], str))


class TestBinaryFormat(unittest.TestCase):

    def test_datum(self):
        bf = index.BinaryFormat.default()
        self.assertEqual(0, bf.decode_datum(bf.encode_datum(0)))
        self.assertEqual(111, bf.decode_datum(bf.encode_datum(111)))
        self.assertEqual(
            bf.max_datum_value,
            bf.decode_datum(bf.encode_datum(bf.max_datum_value)))

    def test_time_interval(self):
        bf = index.BinaryFormat.default()
        self.assertEqual((0, 0), bf.decode_time_interval(
                         bf.encode_time_interval(0, 0)))
        self.assertEqual((0, 100), bf.decode_time_interval(
                         bf.encode_time_interval(0, 100)))
        self.assertEqual((777, 888), bf.decode_time_interval(
                         bf.encode_time_interval(777, 888)))
        self.assertEqual((76543210, 76543210 + bf.max_time_interval),
                         bf.decode_time_interval(bf.encode_time_interval(
                            76543210, 76543210 + bf.max_time_interval)))


class TestInvertedIndex(unittest.TestCase):

    def test_index(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        idx_path = os.path.join(idx_dir, 'index.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with index.InvertedIndex(idx_path, lexicon, documents) as inv_index:
            # Unigram search
            r = inv_index.search('THE')
            for i, d in enumerate(r.documents):
                self.assertEqual(d.count, len(list(d.locations)))
            self.assertEqual(i + 1, r.count)

            # Bigram search
            r = inv_index.search('UNITED STATES')
            for d in r.documents:
                for l in d.locations:
                    self.assertEqual(l.max_index - l.min_index, 1)

            # N-gram search
            r = inv_index.search('UNITED STATES OF AMERICA')
            for d in r.documents:
                for l in d.locations:
                    self.assertEqual(l.max_index - l.min_index, 3)


class TestDocumentData(unittest.TestCase):

    def test_token_data(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        data_path = os.path.join(idx_dir, 'docs.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with index.DocumentData(data_path, lexicon, documents) as doc_data:
            for i in range(len(documents)):
                for t in doc_data.tokens(i):
                    pass
                for t in doc_data.tokens(i, decode=True):
                    pass

    def test_time_index(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        data_path = os.path.join(idx_dir, 'docs.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with index.DocumentData(data_path, lexicon, documents) as doc_data:
            for i in range(len(documents)):
                for interval in doc_data.token_intervals(i, 0, 2 ** 16):
                    for t in interval.tokens:
                        pass
                for interval in doc_data.token_intervals(i, 0, 0):
                    for t in interval.tokens:
                        pass
                for interval in doc_data.token_intervals(i, 0, 2 ** 16,
                                                         decode=True):
                    for t in interval.tokens:
                        pass
                for interval in doc_data.token_intervals(i, 0, 0, decode=True):
                    for t in interval.tokens:
                        pass


class TestUtils(unittest.TestCase):

    def test_window(self):
        input = [0, 1, 2, 3]
        self.assertListEqual(
            list(utils.window(input, 2)), [(0, 1), (1, 2), (2, 3)])
        self.assertListEqual(
            list(utils.window(input, 3)), [(0, 1, 2), (1, 2, 3)])

    def test_topic_search(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        idx_path = os.path.join(idx_dir, 'index.bin')
        documents, lexicon = get_docs_and_lex(idx_dir)
        with index.InvertedIndex(idx_path, lexicon, documents) as inv_index:
            r = utils.topic_search(
                ['UNITED STATES', 'AMERICA', 'US'], inv_index)
            for i, d in enumerate(r.documents):
                self.assertEqual(d.count, len(list(d.locations)))


class TestScripts(unittest.TestCase):

    def test_scan(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        scan.main(idx_dir, os.cpu_count(), None)

    def test_search(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        search.main(idx_dir, ['UNITED', 'STATES'], False, 3)

    def test_build_metadata(self):
        idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
        meta_path = os.path.join(idx_dir, 'meta.bin')

        build_metadata.main(idx_dir, True)

        documents, lexicon = get_docs_and_lex(idx_dir)
        with index.MetadataIndex(
                meta_path, documents,
                build_metadata.NLPTagFormat()) as metadata:
            for d in documents:
                self.assertTrue(d.meta_data_offset >= 0)
                for tag in metadata.metadata(d):
                    self.assertTrue(isinstance(tag, str))


if __name__ == '__main__':
    TMP_DIR = tempfile.mkdtemp(suffix=None, prefix='caption-index-unittest-',
                               dir=None)
    try:
        build_test_index(TMP_DIR)
        unittest.main()
    finally:
        shutil.rmtree(TMP_DIR, True)
