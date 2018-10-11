"""
Inverted index for srt files.
"""

import nltk
import mmap
from collections import namedtuple, deque


def tokenize(s):
    return nltk.word_tokenize(s)


class Lexicon(object):

    Word = namedtuple(
        'Word', [
            'id',       # Token id
            'token',    # String representation
            'count',    # Number of occurrences
            'offset'    # Offset into index file
        ])

    def __init__(self, words):
        """List of words, where w.id is the index in the list"""
        assert isinstance(words, list)
        self._words = words
        self._inverse = {w.token: w for i, w in enumerate(self._words)}

    def __iter__(self):
        # Iterate lexicon in id order
        return self._words.__iter__()

    def __getitem__(self, key):
        if isinstance(key, int):
            # Get word by id (IndexError)
            return self._words[key]
        elif isinstance(key, str):
            # Get word by token (KeyError)
            return self._inverse[key]
        raise TypeError('Not supported for {}'.format(type(key)))

    def __contains__(self, key):
        try:
            self.__getitem__(key)
        except (KeyError, IndexError):
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
        if isinstance(key, int):
            # Get doc name by id (IndexError)
            return self._docs[key]
        elif isinstance(key, str):
            # Get doc id by name (KeyError)
            try:
                return self._docs.index(key)
            except ValueError as e:
                raise KeyError(e)
        raise TypeError('Not supported for {}'.format(type(key)))

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

# Time interval encoding
START_TIME_SIZE = 4
END_TIME_SIZE = 2
TIME_INT_SIZE = START_TIME_SIZE + END_TIME_SIZE
MAX_TIME_INT_VALUE = 2 ** (8 * (START_TIME_SIZE - END_TIME_SIZE)) - 1


def encode_time_int(start, end):
    assert isinstance(start, int)
    assert isinstance(end, int)
    diff = end - start
    if diff < 0:
        raise ValueError('start cannot exceed end: {} > {}'.format(start, end))
    if diff > MAX_TIME_INT_VALUE:
        raise ValueError('end - start > {}'.format(MAX_TIME_INT_VALUE))
    return (start).to_bytes(START_TIME_SIZE, ENDIAN) + (diff).to_bytes(END_TIME_SIZE, ENDIAN)


def decode_time_int(s):
    assert len(s) == TIME_INT_SIZE
    start = int.from_bytes(s[:START_TIME_SIZE], ENDIAN)
    diff = int.from_bytes(s[START_TIME_SIZE:], ENDIAN)
    return start, start + diff


def mmap_decode_time_int(mm, i):
    return decode_time_int(mm[i:i + TIME_INT_SIZE])


# Everything except time intervals are datums
DATUM_SIZE = 3
MAX_DATUM_VALUE = 2 ** (DATUM_SIZE * 8) - 1


def encode_datum(i):
    assert isinstance(i, int)
    if i < 0:
        raise ValueError('Out of range: {} < 0'.format(i))
    if i > MAX_DATUM_VALUE:
        raise ValueError('Out of range: {} > {}'.format(i, MAX_DATUM_VALUE))
    return (i).to_bytes(DATUM_SIZE, ENDIAN)


def decode_datum(s):
    assert len(s) == DATUM_SIZE, '{} is too short'.format(len(s))
    return int.from_bytes(s, ENDIAN)


def mmap_decode_datum(mm, i):
    return decode_datum(mm[i:i + DATUM_SIZE])


def millis_to_seconds(t):
    return t / 1000


def empty_generator():
    return
    yield


def sequence_to_generator(seq):
    for s in seq:
        yield s


class InvertedIndex(object):

    Location = namedtuple(
        'Location', [
            'index',    # Position in document
            'start',    # Start time in seconds
            'end'       # End time in seconds
        ])

    Document = namedtuple(
        'Document', [
            'id',           # Document ID
            'count',        # Number of locations
            'locations'     # Generator of locations
        ])

    Result = namedtuple(
        'Result', [
            'count',        # Count of documents (None if this count is not
                            # available)
            'documents'     # Generator of documents
        ])

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
        if isinstance(text, str):
            tokens = tokenize(text.strip())
            if len(tokens) == 0:
                raise ValueError('No words in input')
            for t in tokens:
                if t not in self._lexicon:
                    raise ValueError('{} is not in the lexicon'.format(t))
        elif isinstance(list):
            tokens = text
            if len(tokens) == 0:
                raise ValueError('No words in input')
        return self.ngram_search(*tokens)

    # TODO: there are more optimized ways to do this
    def ngram_search(self, first_word, *other_words):
        partial_result = self.unigram_search(first_word)
        for i, next_word in enumerate(other_words):
            next_result = self.unigram_search(next_word)
            partial_result = InvertedIndex.Result(
                count=None,
                documents=InvertedIndex._merge_results(
                    partial_result, next_result, i + 1))
        return partial_result

    def _datum_at(self, i):
        return mmap_decode_datum(self._mmap, i)

    def _time_int_at(self, i):
        return mmap_decode_time_int(self._mmap, i)

    def _get_locations(self, offset, count):
        for _ in range(count):
            index = self._datum_at(offset)
            offset += DATUM_SIZE
            start, end = self._time_int_at(offset)
            offset += TIME_INT_SIZE
            yield InvertedIndex.Location(
                index, millis_to_seconds(start),
                millis_to_seconds(end))

    def _get_documents(self, offset, count):
        prev_doc_id = None
        for _ in range(count):
            doc_id = self._datum_at(offset)
            assert prev_doc_id is None or doc_id > prev_doc_id, \
                'Uh oh... document ids should be ascending, but {} <= {}'.format(
                    doc_id, prev_doc_id)
            offset += DATUM_SIZE

            posting_count = self._datum_at(offset)
            assert posting_count > 0, 'Expected at least one posting'
            offset += DATUM_SIZE

            yield InvertedIndex.Document(
                id=doc_id, count=posting_count,
                locations=self._get_locations(offset, posting_count))
            offset += posting_count * (DATUM_SIZE + TIME_INT_SIZE)
            prev_doc_id = doc_id

    def unigram_search(self, word):
        if isinstance(word, Lexicon.Word):
            word = self._lexicon[word.id]
        else:
            word = self._lexicon[word]

        if word.offset < 0:
            return InvertedIndex.Result(count=0, documents=empty_generator())
        assert word.offset < self._mmap.size(), \
            'Offset exceeds file length: {} > {}'.format(
            word.offset, self._mmap.size())

        curr_offset = word.offset

        word_id = self._datum_at(curr_offset)
        assert word_id == word.id, \
            'Expected word id {}, got {}'.format(word.id, word_id)
        curr_offset += DATUM_SIZE

        doc_count = self._datum_at(curr_offset)
        assert doc_count > 0, 'Expected at least one document'
        assert doc_count < len(self._documents), \
            'Uh oh... too many documents: {}'.format(doc_count)
        curr_offset += DATUM_SIZE

        return InvertedIndex.Result(
            count=doc_count,
            documents=self._get_documents(curr_offset, doc_count))

    @staticmethod
    def _merge_results(a, b, gap):
        """Generator for merged results"""
        a_head = next(a.documents)
        b_head = next(b.documents)
        while True:
            if a_head.id == b_head.id:
                # Merge within a document
                locations = None
                try:
                    a_ent = next(a_head.locations)
                    b_ent = next(b_head.locations)
                    while True:
                        if a_ent.index + gap == b_ent.index:
                            if locations is None:
                                locations = deque()
                            locations.append(InvertedIndex.Location(
                                a_ent.index, a_ent.start, b_ent.end))
                            a_ent = next(a_head.locations)
                            b_ent = next(b_head.locations)
                        elif a_ent.index + gap < b_ent.index:
                            a_ent = next(a_head.locations)
                        else:
                            b_ent = next(b_head.locations)
                except StopIteration:
                    pass
                finally:
                    if locations is not None:
                        yield InvertedIndex.Document(
                            id=a_head.id, count=len(locations),
                            locations=sequence_to_generator(locations))
                a_head = next(a.documents)
                b_head = next(b.documents)
            elif a_head.id < b_head.id:
                a_head = next(a.documents)
            else:
                b_head = next(b.documents)
