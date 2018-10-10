"""
Inverted index for srt files.
"""

import nltk
import mmap
from collections import namedtuple, deque


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
    ofs = base_ofs + i * DATUM_SIZE
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
            partial_result = InvertedIndex._merge_results(
                partial_result, next_result, i + 1)
            if len(partial_result) == 0:
                break
        return partial_result

    def unigram_search(self, word):
        if isinstance(word, Lexicon.Word):
            word = self._lexicon[word.id]
        else:
            word = self._lexicon[word]
        
        if word.offset < 0:
            return deque()
        assert word.offset < self._mmap.size(), \
            'Offset exceeds file length: {} > {}'.format(word.offset, self._mmap.size())

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

        result = deque()

        prev_doc_id = None
        for _ in range(doc_count):
            data_idx += 1
            doc_id = mm_read(data_idx)
            assert prev_doc_id is None or doc_id > prev_doc_id, \
                'Uh oh... document ids should be ascending, but {} <= {}'.format(doc_id, prev_doc_id)

            data_idx += 1
            posting_count = mm_read(data_idx)
            assert posting_count > 0, 'Expected at least one posting'

            d = InvertedIndex.Document(id=doc_id, entries=deque())
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

    @staticmethod
    def _merge_results(a, b, gap):
        """Overlap sorted results for ngrams"""
        result = deque()
        while len(a) > 0 and len(b) > 0:
            if a[0].id == b[0].id:
                # Merge within a document
                d = None
                while len(a[0].entries) > 0 and len(b[0].entries) > 0:
                    a_ent = a[0].entries[0]
                    b_ent = b[0].entries[0]
                    if a_ent.position + gap == b_ent.position:
                        if d is None:
                            d = InvertedIndex.Document(id=a[0].id, entries=deque())
                        d.entries.append(InvertedIndex.Entry(
                            a_ent.position, a_ent.start, b_ent.end))
                    elif a_ent.position + gap < b_ent.position:
                        a[0].entries.popleft()
                    else:
                        b[0].entries.popleft()
                if d is not None:
                    result.append(d)
            elif a[0].id < b[0].id:
                a.popleft()
            else:
                b.popleft()
        return result
