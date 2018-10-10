"""
Inverted index for srt files.
"""

import nltk
import mmap
from collections import namedtuple


def tokenize(s):
    return nltk.word_tokenize(s)


class Lexicon(object):

    Word = namedtuple('Word', ['id', 'token', 'count', 'offset'])

    def __init__(self, words):
        """List of words, where w.id is the index in the list"""
        assert isinstance(words, list)
        self._words = words
        self._inverse = {w.token: w for i, w in enumerate(self._words)}

    def __iter__(self):
        # Iterate lexicon in id order
        return self._words.__iter__()

    def __getitem__(self, key):
        try:
            if isinstance(key, int):
                # Get word by id
                return self._words[key]
            elif isinstance(key, str):
                # Get word by token
                return self._inverse[key]
        except (IndexError, KeyError) as e:
            raise KeyError(e)
        raise NotImplementedError('Not supported for {}'.format(type(key)))

    def __contains__(self, key):
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __len__(self):
        return len(self._words)

    def store(self, path):
        with open(path, 'w') as f:
            for w in self._words:
                f.write('{}\t{}\t{}\n'.format(w.token, w.count, w.offset))

    @staticmethod
    def load(path):
        words = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                token, count, offset = line.strip().split('\t')
                count = int(count)
                offset = int(offset)
                words.append(Lexicon.Word(i, token, count, offset))
        return Lexicon(words)


class Documents(object):

    def __init__(self, docs):
        """List of document names, where index is the id"""
        self._docs = docs

    def __iter__(self):
        return self._docs.__iter__()

    def __getitem__(self, key):
        try:
            if isinstance(key, int):
                # Get doc name by id
                return self._docs[key]
            elif isinstance(key, str):
                # Get doc id by name
                index = self._docs.index(key)
                if index < 0:
                    raise KeyError('no document with name {}'.format(key))
                return index
        except (KeyError, IndexError) as e:
            raise KeyError(e)
        raise NotImplementedError('Not supported for {}'.format(type(key)))

    def __contains__(self, key):
        try:
            self.__getitem__(key)
        except (KeyError, IndexError):
            return False
        return True

    def __len__(self):
        return len(self._docs)

    def store(self, path):
        with open(path, 'w') as f:
            for d in self._docs:
                f.write(d)
                f.write('\n')

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            return Documents([d.strip() for d in f])


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


def mmap_decode_datum(mm, base_ofs, i=0):
    ofs = base_ofs + i
    return decode_datum(mm[ofs:ofs + DATUM_SIZE])


def millis_to_seconds(t):
    return t / 1000


class InvertedIndex(object):

    Entry = namedtuple('Entry', ['position', 'start', 'end'])
    Document = namedtuple('Document', ['id', 'entries'])

    def __init__(self, path, lexicon, documents):
        assert isinstance(lexicon, Lexicon)
        assert isinstance(documents, Documents)
        assert isinstance(path, str)
        self._lexicon = lexicon
        self._documents = documents
        self._f = open(path, 'rb')
        self._mmap = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        if self._f is not None:
            self._mmap.close()
            self._mmap = None
            self._f.close()
            self._f = None

    def search(self, text):
        raise NotImplementedError()

    def unigram_search(self, key):
        if isinstance(key, Lexicon.Word):
            word = self._lexicon[key.id]
        else:
            word = self._lexicon[key]
        if word.offset < 0:
            return []
        assert word.offset < self._mmap.size(), \
            'Offset exceeds file length: {}'.format(word.offset)

        base_offset = word.offset

        def mm_read(i):
            return mmap_decode_datum(self._mmap, base_offset, i)

        data_idx = 0
        word_id = mm_read(data_idx)
        assert word_id == word.id, \
            'Expected word id {}, got {}'.format(word.id, word_id)

        data_idx += 1
        doc_count = mm_read(data_idx)
        assert doc_count > 0, 'Expected at least one document'
        assert doc_count < len(self._documents), 'Uh oh... too many documents: {}'.format(doc_count)

        result = []

        prev_doc_id = None
        for _ in range(doc_count):
            data_idx += 1
            doc_id = mm_read(data_idx)
            assert prev_doc_id is None or doc_id > prev_doc_id, \
                'Uh oh... document ids should be ascending, but {} <= {}'.format(doc_id, prev_doc_id)

            data_idx += 1
            posting_count = mm_read(data_idx)
            assert posting_count > 0, 'Expected at least one posting'

            d = InvertedIndex.Document(id=doc_id, entries=[])
            for _ in range(posting_count):
                data_idx += 1
                position = mm_read(data_idx)
                data_idx += 1
                start = millis_to_seconds(mm_read(data_idx))
                data_idx += 1
                end = millis_to_seconds(mm_read(data_idx))
                d.entries.append(InvertedIndex.Entry(position, start, end))
            result.append(d)
            prev_doc_id = doc_id
        return result

    def ngram_search(self, keys):
        raise NotImplementedError()
