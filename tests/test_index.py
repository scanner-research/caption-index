"""
Build a dummy index and run tests on it.
"""

import os
import pytest
import shutil
import tempfile
from subprocess import check_call

import captions as captions
import captions.decode as decode

from lib.common import get_docs_and_lexicon

TMP_DIR = None
TEST_SUBS_SUBDIR = 'subs'
TEST_INDEX_SUBDIR = 'index'
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test-small.tar.gz')

BUILD_INDEX_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scripts', 'build_index.py')


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

        # Build the index
        check_call([
            BUILD_INDEX_SCRIPT, '-d', subs_dir, '-o', idx_dir,
            '--keep-tmp-files'])
        assert os.path.isdir(idx_dir)

    try:
        build_test_index(TMP_DIR)
        yield
    finally:
        shutil.rmtree(TMP_DIR, True)


def test_search():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = get_docs_and_lexicon(idx_dir)

    def count_and_test(index, document, tokens):
        ids = index.contains(tokens, [document])
        assert len(ids) == 1

        count = 0
        (d,) = list(index.search(tokens, [document]))
        assert len(d.postings) > 0
        dh = documents.open(document)
        for l in d.postings:
            assert l.len == len(tokens)
            assert abs(l.end - l.start) < 10.0, 'ngram time too large'
            count += 1

            # Check that we actually found the right ngrams
            assert [lexicon.decode(t) for t in dh.tokens(l.idx, l.len)] == tokens

        return count

    test_document = documents['cnn.srt']

    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        assert count_and_test(index, test_document, ['THEY']) == 12
        assert count_and_test(index, test_document, ['PEOPLE']) == 12
        assert count_and_test(index, test_document, ['TO', 'THE']) == 9    # one wraps
        assert count_and_test(index, test_document, ['GIBSON', 'GUITAR', 'DROP']) == 1
        assert count_and_test(index, test_document, ['PUT', 'THAT', 'DOWN']) == 1
        assert count_and_test(index, test_document, ['CLOCK', 'STRIKES']) == 2
        assert count_and_test(index, test_document, ['>>']) == 149
        assert count_and_test(index, test_document, ['SEE', '?']) == 1

    # Make a chunked copy
    chunked_idx_path = os.path.join(idx_dir, 'index.tmp')
    assert len(os.listdir(chunked_idx_path)) > 0
    with captions.CaptionIndex(chunked_idx_path, lexicon, documents) as index2:
        assert count_and_test(index2, test_document, ['THEY']) == 12
        assert count_and_test(index2, test_document, ['PEOPLE']) == 12
        assert count_and_test(index2, test_document, ['TO', 'THE']) == 9    # one wraps
        assert count_and_test(index2, test_document, ['GIBSON', 'GUITAR', 'DROP']) == 1
        assert count_and_test(index2, test_document, ['PUT', 'THAT', 'DOWN']) == 1
        assert count_and_test(index2, test_document, ['CLOCK', 'STRIKES']) == 2
        assert count_and_test(index2, test_document, ['>>']) == 149
        assert count_and_test(index2, test_document, ['SEE', '?']) == 1
    shutil.rmtree(chunked_idx_path)


def test_search_position():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = get_docs_and_lexicon(idx_dir)

    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        test_document = documents['test.srt']
        dh = documents.open(test_document)

        # In range
        for i in range(10):
            assert dh.position(5 * i + 2.5) == i

        # Out of range
        assert dh.position(51) == 10
        assert dh.position(100) == 10


def _is_close(a, b):
    return abs(a - b) <= 1e-6


def test_search_intervals():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = get_docs_and_lexicon(idx_dir)

    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        test_document = documents['test.srt']

        # unigrams
        for i in range(10):
            (d,) = list(index.search([str(i + 1)], [test_document]))
            (p,) = d.postings
            assert _is_close(p.start, i * 5.)
            assert _is_close(p.end, (i + 1) * 5.)

        # bigrams
        for i in range(9):
            bigram = [str(i + 1), str(i + 2)]
            (d,) = list(index.search(bigram, [test_document]))
            (p,) = d.postings
            assert _is_close(p.start, i * 5.), bigram
            assert _is_close(p.end, (i + 2) * 5.), bigram

        # 3-grams
        for i in range(8):
            trigram = [str(i + 1), str(i + 2), str(i + 3)]
            (d,) = list(index.search(trigram, [test_document]))
            (p,) = d.postings
            assert _is_close(p.start, i * 5.), trigram
            assert _is_close(p.end, (i + 3) * 5.), trigram


def test_decode():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    documents, lexicon = get_docs_and_lexicon(idx_dir)
    doc_handle = documents.open(0)
    print(decode.get_vtt(lexicon, doc_handle))
    print(decode.get_srt(lexicon, doc_handle))
