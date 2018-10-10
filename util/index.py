import nltk
from collections import namedtuple


def tokenize(s):
    return nltk.word_tokenize(s)


class Lexicon(object):

    Word = namedtuple('Word', ['id', 'token', 'count', 'offset'])

    def __init__(self, words):
        assert isinstance(words, list)
        self._words = words
        self._inverse = {w.token: w for i, w in enumerate(self._words)}

    def __iter__(self):
        return self._words.__iter__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._words[key]
        elif isinstance(key, str):
            return self._inverse.get(key, None)
        raise KeyError()

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
        self._docs = docs

    def __iter__(self):
        return self._docs.__iter__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._docs[key]
        elif isinstance(key, str):
            return self._docs.index(key)
        raise KeyError()

    def __contains__(self, key):
        try:
            self.__getitem__(key)
        except KeyError:
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
