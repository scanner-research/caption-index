from typing import Set

PARTS_OF_SPEECH = ['noun', 'verb']


class SpacyLemmatizer(object):

    def __init__(self):
        from spacy.lemmatizer import Lemmatizer
        from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

        self._lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    def lemma(self, token: str) -> Set[str]:
        results = set()
        for pos in PARTS_OF_SPEECH:
            results.update(self._lemmatizer(token, pos))
        return results


def default_lemmatizer() -> SpacyLemmatizer:
    return SpacyLemmatizer()
