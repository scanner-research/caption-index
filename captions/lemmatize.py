from abc import ABC, abstractmethod
from typing import Set


class Lemmatizer(ABC):

    @abstractmethod
    def lemma(self, token: str) -> Set[str]:
        pass


class SpacyLemmatizer(Lemmatizer):
    # Requres en
    # python -m spacy download en_core_web_sm

    def __init__(self):
        import spacy
        self._nlp = spacy.load('en_core_web_sm', 
                               disable=['parser', 'ner'])

    def lemma(self, token: str) -> Set[str]:
        return {t.lemma_ for t in self._nlp(token)}


class WordNetLemmatizer(Lemmatizer):

    def __init__(self):
        import nltk
        from nltk.stem import WordNetLemmatizer 
        
        self._lemmatizer = WordNetLemmatizer()
        try:
            assert list(self.lemma('feet'))[0] == 'foot'
        except LookupError:
            nltk.download('wordnet')

    def lemma(self, token: str) -> Set[str]:
        return {self._lemmatizer.lemmatize(token),}


def default_lemmatizer() -> Lemmatizer:
    return WordNetLemmatizer()
