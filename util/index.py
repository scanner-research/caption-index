import _pickle as pickle
import nltk


UNKNOWN_TOKEN_ID = -1


def tokenize(s):
    return nltk.word_tokenize(s)


class DocumentIndex(object):

    def __init__(self, index, doclist, lexicon):
        self._index = index
        self._doclist = doclist
        self._lexicon = lexicon

    def search(self, s):
        pass

    def store(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'index': self._index,
                'doclist': self._doclist,
                'lexicon': self._lexicon
            }, f, protocol=-1)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return DocumentIndex(**pickle.load(f))
