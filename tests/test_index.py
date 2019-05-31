"""
Build a dummy index and run tests on it.
"""

import os
import pytest
import sys
import shutil
import tempfile
from subprocess import check_call

import captions as captions


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../scripts')

import build


TMP_DIR = None
TEST_SUBS_SUBDIR = 'subs'
TEST_INDEX_SUBDIR = 'index'
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test-small.tar.gz')


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


def test_search():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)

    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        test_document = documents['cnn.srt']

        def count_and_test(tokens):
            ids = index.contains(tokens, [test_document])
            assert len(ids) == 1

            count = 0
            (d,) = list(index.search(tokens, [test_document]))
            assert len(d.postings) > 0
            for l in d.postings:
                assert l.len == len(tokens)
                assert abs(l.end - l.start) < 10.0, 'ngram time too large'
                count += 1

                # Check that we actually found the right ngrams
                assert [
                    lexicon.decode(t) for t in
                    index.tokens(d.id, l.idx, l.len)
                ] == tokens

            return count

        assert count_and_test(['THEY']) == 12
        assert count_and_test(['PEOPLE']) == 12
        assert count_and_test(['TO', 'THE']) == 9    # one wraps
        assert count_and_test(['GIBSON', 'GUITAR', 'DROP']) == 1
        assert count_and_test(['PUT', 'THAT', 'DOWN']) == 1
        assert count_and_test(['CLOCK', 'STRIKES']) == 2
        assert count_and_test(['>>']) == 149
        assert count_and_test(['SEE', '?']) == 1


def test_search_position():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)

    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        test_document = documents['test.srt']

        # In range
        for i in range(10):
            assert index.position(test_document, 5 * i + 2.5) == i

        # Out of range
        assert index.position(test_document, 51) == 10
        assert index.position(test_document, 100) == 10


def _is_close(a, b):
    return abs(a - b) <= 1e-6


def test_search_intervals():
    idx_dir = os.path.join(TMP_DIR, TEST_INDEX_SUBDIR)
    idx_path = os.path.join(idx_dir, 'index.bin')
    documents, lexicon = _get_docs_and_lex(idx_dir)

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
