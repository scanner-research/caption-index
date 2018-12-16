#!/usr/bin/env python3

import os
import pytest
import sys
import shutil
import tempfile
from subprocess import check_call

import captions as captions
import captions.util as util

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../scripts')

import build
import scan
import search
import lexicon as pmi_lexicon
import build_metadata
import build_ngrams


TMP_DIR = None
TEST_SUBS_SUBDIR = 'subs'
TEST_INDEX_SUBDIR = 'index'
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test.tar.gz')


@pytest.fixture(scope="session", autouse=True)
def dummy_data():
    global TMP_DIR
    TMP_DIR = tempfile.mkdtemp(suffix=None, prefix='caption-index-unittest-',
                               dir=None)

    def build_test_index(tmp_dir):
        subs_dir = os.path.join(tmp_dir, TEST_SUBS_SUBDIR)
        idx_dir = os.path.join(tmp_dir, TEST_INDEX_SUBDIR)

        # Unpack the test data
        os.makedirs(subs_dir)
        check_call(['tar', '-xzf', TEST_DATA_PATH, '-C', subs_dir])

        build.main(subs_dir, idx_dir, 1)
        assert os.path.isdir(idx_dir)

    try:
        build_test_index(TMP_DIR)
        yield
    finally:
        shutil.rmtree(TMP_DIR, True)


def _get_docs_and_lex(idx_dir):
    doc_path = os.path.join(idx_dir, 'docs.list')
    lex_path = os.path.join(idx_dir, 'words.lex')

    documents = captions.Documents.load(doc_path)
    lexicon = captions.Lexicon.load(lex_path)
    return documents, lexicon


def test_tokenize():
    text = 'I\'m a string! This is a tokenizer test.'
    tokens = list(captions.default_tokenizer().tokens(text))
    assert isinstance(tokens[0], str)
    assert len(tokens) == 11


def test_binary_format():
    bf = captions.BinaryFormat.default()
    assert 0 == bf._decode_datum(bf.encode_datum(0))
    assert 111 == bf._decode_datum(bf.encode_datum(111))
    assert bf.max_datum_value == \
        bf._decode_datum(bf.encode_datum(bf.max_datum_value))

    assert 0 == bf._decode_u32(bf.encode_u32(0))
    assert 111 == bf._decode_u32(bf.encode_u32(111))
    assert bf.max_datum_value == \
        bf._decode_u32(bf.encode_u32(bf.max_datum_value))

    assert (0, 0) == bf._decode_time_interval(bf.encode_time_interval(0, 0))
    assert (0, 100) == bf._decode_time_interval(bf.encode_time_interval(0, 100))
    assert (777, 888) == bf._decode_time_interval(
        bf.encode_time_interval(777, 888))
    assert (76543210, 76543210 + bf.max_time_interval) == \
        bf._decode_time_interval(bf.encode_time_interval(
            76543210, 76543210 + bf.max_time_interval))


def test_inverted_index():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)

    def test_search_and_contains(tokens):
        ids = index.contains(tokens)
        search_ids = set()
        for d in index.search(tokens):
            assert len(d.postings) > 0
            for l in d.postings:
                assert l.len == len(tokens)
            search_ids.add(d.id)
        assert ids == search_ids

    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        # Unigram search
        test_search_and_contains(['THE'])
        test_search_and_contains(['UNITED'])
        test_search_and_contains(['STATES'])
        test_search_and_contains(['AND'])

        # Bigram search
        test_search_and_contains(['UNITED', 'STATES'])
        test_search_and_contains(['UNITED', 'KINGDOM'])

        # N-gram search
        test_search_and_contains(['UNITED', 'STATES', 'OF', 'AMERICA'])


def test_token_data():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)
    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        for i in range(len(documents)):
            for t in index.tokens(i):
                lexicon.decode(t)


def test_intervals_data():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)
    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        for i in range(len(documents)):
            assert len(index.intervals(i, 0, 0)) == 0
            for posting in index.intervals(i, 0, 2 ** 16):
                pass


def test_util_window():
    input = [0, 1, 2, 3]
    assert list(util.window(input, 2)) == [(0, 1), (1, 2), (2, 3)]
    assert list(util.window(input, 3)) == [(0, 1, 2), (1, 2, 3)]


def test_frequent_words():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    _, lexicon = _get_docs_and_lex(idx_dir)
    assert len(util.frequent_words(lexicon, 100)) == 1
    assert len(util.frequent_words(lexicon, 0)) == len(lexicon)
    assert len(util.frequent_words(lexicon, 99)) > 0


def test_topic_search():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)
    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        for d in util.topic_search(['UNITED STATES', 'AMERICA', 'US'], index):
            assert len(d.postings) > 0


def test_script_scan():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    scan.main(idx_dir, os.cpu_count(), None)


def test_script_search():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    search.main(idx_dir, ['UNITED', 'STATES'], False, 3)


def test_script_build_metadata():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    meta_path = os.path.join(idx_dir, 'meta.bin')

    build_metadata.main(idx_dir, True)

    documents, lexicon = _get_docs_and_lex(idx_dir)
    with captions.MetadataIndex(
            meta_path, documents,
            build_metadata.NLPTagFormat()) as metadata:
        for d in documents:
            assert len(metadata.metadata(d, 0, 0)) == 0
            for tag in metadata.metadata(d):
                assert isinstance(tag, str)


def test_script_build_ngrams_and_lexicon():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    ngram_path = os.path.join(idx_dir, 'ngrams.bin')

    build_ngrams.main(idx_dir, n=5, min_count=10, workers=os.cpu_count(),
                      limit=None)

    _, lexicon = _get_docs_and_lex(idx_dir)
    ngram_frequency = captions.NgramFrequency(ngram_path, lexicon)

    def test_phrase(tokens):
        assert ' '.join(tokens) in ngram_frequency
        assert ngram_frequency[' '.join(tokens)] > 0
        assert ngram_frequency[tokens] > 0
        ids = tuple(lexicon[t].id for t in tokens)
        assert ngram_frequency[ids] > 0

    test_phrase(('UNITED',))
    test_phrase(('UNITED', 'STATES'))
    test_phrase(('THE', 'UNITED', 'STATES'))
    test_phrase(('OF', 'THE', 'UNITED', 'STATES'))
    test_phrase(('PRESIDENT', 'OF', 'THE', 'UNITED', 'STATES'))

    pmi_lexicon.main(idx_dir, ['UNITED', 'STATES'], 5, 30, 10)
