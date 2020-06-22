"""
Build a dummy index, update it, and run tests on it.
"""

import os
import shutil
import tempfile
from subprocess import check_call, CalledProcessError

import captions as captions
from lib.common import get_docs_and_lexicon

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test-small.tar.gz')

BUILD_INDEX_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scripts', 'build_index.py')
UPDATE_INDEX_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scripts', 'update_index.py')


def test_update_index():
    tmp_dir = tempfile.mkdtemp(suffix=None, prefix='caption-index-unittest-',
                               dir=None)
    subs_dir = os.path.join(tmp_dir, 'subs')
    idx_dir = os.path.join(tmp_dir, 'index')

    # Unpack the test data
    os.makedirs(subs_dir)
    check_call(['tar', '-xzf', TEST_DATA_PATH, '-C', subs_dir])

    # Build an index
    check_call([BUILD_INDEX_SCRIPT, '-d', subs_dir, '-o', idx_dir])

    # Update the index (should fail due to duplicate files)
    try:
        check_call([UPDATE_INDEX_SCRIPT, '-d', subs_dir, idx_dir])
        raise Exception('Uh oh, an exception should have been thrown...')
    except CalledProcessError:
        pass

    # Update the index (should do nothing since all of them are duplicates)
    check_call([UPDATE_INDEX_SCRIPT, '--skip-existing-names', '-d', subs_dir,
                idx_dir])

    # Update the index
    for fname in os.listdir(subs_dir):
        src_path = os.path.join(subs_dir, fname)
        dst_path = os.path.join(subs_dir, 'copy::' + fname)
        shutil.move(src_path, dst_path)
    check_call([UPDATE_INDEX_SCRIPT, '-d', subs_dir, idx_dir])
    assert os.path.isfile(os.path.join(idx_dir, 'documents.txt.old'))

    # Test the new index
    def count_and_test(index, document, tokens):
        ids = index.contains(tokens, [document])
        assert len(ids) == 1

        count = 0
        (d,) = list(index.search(tokens, [document]))
        dh = documents.open(document)
        assert len(d.postings) > 0
        for l in d.postings:
            assert l.len == len(tokens)
            assert abs(l.end - l.start) < 10.0, 'ngram time too large'
            count += 1

            # Check that we actually found the right ngrams
            assert [lexicon.decode(t) for t in dh.tokens(l.idx, l.len)] == tokens

        return count

    documents, lexicon = get_docs_and_lexicon(idx_dir)
    idx_path = os.path.join(idx_dir, 'index.bin')
    assert os.path.isdir(idx_path)
    assert len(os.listdir(idx_path)) == 2, os.listdir(idx_path)

    test_document = documents['copy::cnn.srt']
    with captions.CaptionIndex(idx_path, lexicon, documents) as index:
        assert count_and_test(index, test_document, ['THEY']) == 12
        assert count_and_test(index, test_document, ['PEOPLE']) == 12
        assert count_and_test(index, test_document, ['TO', 'THE']) == 9    # one wraps
        assert count_and_test(index, test_document, ['GIBSON', 'GUITAR', 'DROP']) == 1
        assert count_and_test(index, test_document, ['PUT', 'THAT', 'DOWN']) == 1
        assert count_and_test(index, test_document, ['CLOCK', 'STRIKES']) == 2
        assert count_and_test(index, test_document, ['>>']) == 149
        assert count_and_test(index, test_document, ['SEE', '?']) == 1
