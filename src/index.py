"""
Indexes for srt files
"""

import mmap
from collections import namedtuple, deque

import nlp


class Lexicon(object):
    """A map from word to id, and vice versa"""

    UNKNOWN_TOKEN = '<UNKNOWN>'

    Word = namedtuple(
        'Word', [
            'id',       # Token id
            'token',    # String representation
            'count',    # Number of occurrences
            'offset'    # Offset into inverted index file
        ])

    def __init__(self, words):
        """List of words, where w.id is the index in the list"""
        assert isinstance(words, list)
        assert all(i == w.id for i, w in enumerate(words))
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
    """A mapping from document id to name, and vice versa"""

    Document = namedtuple(
        'Document', [
            'id',
            'name',                 # File name
            'length',               # Number of tokens in file
            'time_index_offset',    # Time index offset in binary docs file
            'token_data_offset'     # Token data offset in binary docs file
        ])

    def __init__(self, docs):
        """List of Documents, where index is the id"""
        assert all(i == d.id for i, d in enumerate(docs))
        self._docs = docs

    def __iter__(self):
        return self._docs.__iter__()

    def __getitem__(self, key):
        if isinstance(key, int):
            # Get doc name by id (IndexError)
            return self._docs[key]
        elif isinstance(key, str):
            # Get doc id by name (KeyError)
            for d in self._docs:
                if d.name == key:
                    return d.id
            else:
                raise KeyError('No document named {}'.format(key))
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
                f.write('{}\t{}\t{}\t{}\n'.format(
                    d.name, d.length, d.time_index_offset, d.token_data_offset
                ))

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            documents = []
            for i, line in enumerate(f):
                name, length, time_index_offset, token_data_offset = \
                    line.strip().split('\t')
                length = int(length)
                time_index_offset = int(time_index_offset)
                token_data_offset = int(token_data_offset)
                documents.append(Documents.Document(
                    id=i, name=name, length=length,
                    time_index_offset=time_index_offset,
                    token_data_offset=token_data_offset))
            return Documents(documents)


class BinaryFormat(object):
    """
    Binary data formatter for writing and reading the indexes

    Supports two data types:
        - time interval
        - datum
    """

    Config = namedtuple(
        'Config', [
            'endian',
            'start_time_bytes',     # Number of bytes to encode start times
            'end_time_bytes',       # Number of bytes to encode end - start
            'datum_bytes',          # Number of bytes to encode other data
        ])

    def __init__(self, config):
        assert config.endian in ['big', 'little']
        self._endian = config.endian

        assert config.start_time_bytes > 0
        assert config.end_time_bytes > 0
        self._start_time_bytes = config.start_time_bytes
        self._end_time_bytes = config.end_time_bytes

        assert config.datum_bytes > 0
        self._datum_bytes = config.datum_bytes

        # Derived values
        self._time_interval_bytes = (
            config.start_time_bytes + config.end_time_bytes)
        self._max_time_interval = (
            2 ** (8 * (config.start_time_bytes - config.end_time_bytes)) - 1)
        self._max_datum_value = 2 ** (config.datum_bytes * 8) - 1

    @property
    def time_interval_bytes(self):
        return self._time_interval_bytes

    @property
    def datum_bytes(self):
        return self._datum_bytes

    @property
    def max_time_interval(self):
        """Largest number of milliseconds between start and end times"""
        return self._max_time_interval

    @property
    def max_datum_value(self):
        """Largest value that can be serialized"""
        return self._max_datum_value

    def encode_time_interval(self, start, end):
        assert isinstance(start, int)
        assert isinstance(end, int)
        diff = end - start
        if diff < 0:
            raise ValueError(
                'start cannot exceed end: {} > {}'.format(start, end))
        if diff > self.max_time_interval:
            raise ValueError('end - start > {}'.format(self.max_time_interval))
        return (
            (start).to_bytes(self._start_time_bytes, self._endian) +
            (diff).to_bytes(self._end_time_bytes, self._endian))

    def decode_time_interval(self, s):
        assert len(s) == self.time_interval_bytes
        start = int.from_bytes(s[:self._start_time_bytes], self._endian)
        diff = int.from_bytes(s[self._end_time_bytes:], self._endian)
        return start, start + diff

    def encode_datum(self, i):
        assert isinstance(i, int)
        if i < 0:
            raise ValueError('Out of range: {} < 0'.format(i))
        if i > self._max_datum_value:
            raise ValueError('Out of range: {} > {}'.format(
                             i, self._max_datum_value))
        return (i).to_bytes(self._datum_bytes, self._endian)

    def decode_datum(self, s):
        assert len(s) == self._datum_bytes, '{} is the wrong length'.format(len(s))
        return int.from_bytes(s, self._endian)

    def encode_byte(self, b):
        assert isinstance(b, int)
        assert b <= 255, 'Out of range: {}'.format(b)
        return (b).to_bytes(1, self._endian) # endian arg is silly

    def decode_byte(self, s):
        assert len(s) == 1, '{} is the wrong length'.format(len(s))
        return int.from_bytes(s, self._endian) # endian arg is silly

    @staticmethod
    def default():
        return BinaryFormat(
            BinaryFormat.Config(
                endian='little',
                start_time_bytes=4,
                end_time_bytes=2,
                datum_bytes=3))


def millis_to_seconds(t):
    return t / 1000


def empty_generator():
    return
    yield


class _MemoryMappedFile(object):
    """
    Base class for an object backed by a mmapped file
    """

    def __init__(self, path, binary_format):
        assert isinstance(path, str)

        if binary_format is not None:
            assert isinstance(binary_format, BinaryFormat)
            self._bin_fmt = binary_format
        else:
            self._bin_fmt = BinaryFormat.default()

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

    def _byte_at(self, i):
        return self._bin_fmt.decode_byte(self._mmap[i])

    def _datum_at(self, i):
        return self._bin_fmt.decode_datum(
            self._mmap[i:i + self._bin_fmt.datum_bytes])

    def _time_int_at(self, i):
        return self._bin_fmt.decode_time_interval(
            self._mmap[i:i + self._bin_fmt.time_interval_bytes])


class InvertedIndex(_MemoryMappedFile):
    """
    Interface to a binary encoded inverted index file

    The format of the file is (field, type):

        token_id: datum
        document_count: datum
        |-- document_id: datum
        |-- location_count: datum
        |   |-- location_idx: datum
        |   |-- start, end: time_interval
        |   |-- ... (more locations)
        |-- ... (more documents)

    For O(1) indexing, the Lexicon contains a map of tokens to their binary
    offsets in this file.
    """

    LocationResult = namedtuple(
        'LocationResult', [
            'index',    # Position in document
            'start',    # Start time in seconds
            'end'       # End time in seconds
        ])

    DocumentResult = namedtuple(
        'DocumentResult', [
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

    def __init__(self, path, lexicon, documents, binary_format=None):
        super(InvertedIndex, self).__init__(path, binary_format)

        assert isinstance(lexicon, Lexicon)
        assert isinstance(documents, Documents)
        self._lexicon = lexicon
        self._documents = documents

    def search(self, text):
        if isinstance(text, str):
            tokens = nlp.tokenize(text.strip())
            if len(tokens) == 0:
                raise ValueError('No words in input')
            for t in tokens:
                if t not in self._lexicon:
                    raise ValueError('{} is not in the lexicon'.format(t))
        elif isinstance(text, list):
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

    def _get_locations(self, offset, count):
        for _ in range(count):
            index = self._datum_at(offset)
            offset += self._bin_fmt.datum_bytes
            start, end = self._time_int_at(offset)
            offset += self._bin_fmt.time_interval_bytes
            yield InvertedIndex.LocationResult(
                index, millis_to_seconds(start),
                millis_to_seconds(end))

    def _get_documents(self, offset, count):
        prev_doc_id = None
        for _ in range(count):
            doc_id = self._datum_at(offset)
            assert doc_id < len(self._documents), \
                'Invalid document id: {}'.format(doc_id)
            assert prev_doc_id is None or doc_id > prev_doc_id, \
                'Uh oh... document ids should be ascending, but {} <= {}'.format(
                    doc_id, prev_doc_id)
            offset += self._bin_fmt.datum_bytes

            posting_count = self._datum_at(offset)
            assert posting_count > 0, 'Expected at least one posting'
            offset += self._bin_fmt.datum_bytes

            yield InvertedIndex.DocumentResult(
                id=doc_id, count=posting_count,
                locations=self._get_locations(offset, posting_count))
            offset += posting_count * (
                self._bin_fmt.datum_bytes + self._bin_fmt.time_interval_bytes)
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
        curr_offset += self._bin_fmt.datum_bytes

        doc_count = self._datum_at(curr_offset)
        assert doc_count > 0, 'Expected at least one document'
        assert doc_count <= len(self._documents), \
            'Uh oh... too many documents: {}'.format(doc_count)
        curr_offset += self._bin_fmt.datum_bytes

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
                            locations.append(InvertedIndex.LocationResult(
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
                        yield InvertedIndex.DocumentResult(
                            id=a_head.id, count=len(locations),
                            locations=iter(locations))
                a_head = next(a.documents)
                b_head = next(b.documents)
            elif a_head.id < b_head.id:
                a_head = next(a.documents)
            else:
                b_head = next(b.documents)


class DocumentData(_MemoryMappedFile):
    """
    Interface to a binary encoded document data file

    For each document, there are two sections, always sequential.

    [Time Index]
    Array of the following, repeated for each entry in transcript.

        start, end: time_interval
        position: datum

    [Token Data]
    Array of tokens.

        token: datum
        part_of_speech: byte

    For O(1) indexing, the Documents object contains a map from documents to
    their Time Index and Token Data offsets, and the number of tokens.
    """

    Interval = namedtuple(
        'Interval', [
            'start',        # Start time in seconds
            'end',          # End time in seconds
            'position',     # Start position in document
            'length',       # Number of tokens in interval
            'tokens'        # Generator for tokens in the interval
        ])

    def __init__(self, path, lexicon, documents, binary_format=None):
        super(DocumentData, self).__init__(path, binary_format)

        assert isinstance(lexicon, Lexicon)
        assert isinstance(documents, Documents)
        self._documents = documents
        self._lexicon = lexicon

    def _tokens(self, offset, n, decode):
        entry_bytes = self._bin_fmt.datum_bytes + 1
        for i in range(n):
            token_id = self._datum_at(offset + i * entry_bytes)
            pos_id = self._byte_at(offset + i * entry_bytes + self._bin_fmt.datum_bytes)
            if decode:
                try:
                    token = self._lexicon[token_id].token
                except IndexError:
                    token = Lexicon.UNKNOWN_TOKEN
                yield (token, nlp.POSTag.decode(pos_id))
            else:
                yield (token_id, pos_id)

    def tokens(self, doc, start_pos=None, end_pos=None, decode=False):
        """Generator over tokens in the range (end is non-inclusive)"""
        if isinstance(doc, Documents.Document):
            doc = self._documents[doc.id]
        else:
            doc = self._documents[doc]

        assert doc.length >= 0, 'Invalid document length: {}'.format(doc.length)
        assert doc.token_data_offset >= 0, 'Invalid data offset'

        if end_pos is None or end_pos > doc.length:
            end_pos = doc.length

        if start_pos is None:
            start_pos = 0
        elif start_pos < 0:
            raise ValueError('Start position cannot be negative: {}'.format(
                             start_pos))
        elif start_pos >= end_pos:
            return empty_generator()

        start_offset = (doc.token_data_offset +
                        start_pos * self._bin_fmt.datum_bytes)
        return self._tokens(start_offset, end_pos - start_pos, decode)

    def token_intervals(self, doc, start_time, end_time, decode=False):
        """Generator over transcript intervals and tokens"""
        if isinstance(doc, Documents.Document):
            doc = self._documents[doc.id]
        else:
            doc = self._documents[doc]

        assert doc.length >= 0, 'Invalid document length: {}'.format(doc.length)
        assert doc.time_index_offset >= 0, 'Invalid time index offset'
        assert doc.token_data_offset >= 0, 'Invalid data offset'

        base_idx_offset = doc.time_index_offset
        base_token_offset = doc.token_data_offset
        num_intervals = (doc.token_data_offset - doc.time_index_offset) / (
            self._bin_fmt.time_interval_bytes + self._bin_fmt.datum_bytes)
        for i in range(num_intervals):
            curr_offset = base_idx_offset + i * (
                self._bin_fmt.time_interval_bytes + self._bin_fmt.datum_bytes)
            start, end = self._time_int_at(curr_offset)
            curr_offset + self._bin_fmt.time_interval_bytes
            position = self._datum_at(curr_offset)
            curr_offset += self._bin_fmt.datum_bytes

            if i == num_intervals - 1:
                next_position = doc.length
            else:
                # Peek at next entry, skip time interval
                curr_offset += self._bin_fmt.time_interval_bytes
                next_position = self._datum_at(curr_offset)

            if min(end, end_time) - max(start, start_time) > 0:
                length = next_position - position
                yield DocumentData.Interval(
                    start=start, end=end, position=position, length=length,
                    tokens=self._tokens(
                        (base_token_offset
                            + position * self._bin_fmt.datum_bytes),
                        length, decode
                    ))
