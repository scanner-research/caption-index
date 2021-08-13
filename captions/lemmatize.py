from typing import Set


class SpacyLemmatizer:
    # Requres en
    # python -m spacy download en_core_web_sm

    def __init__(self):
        import spacy
        self._nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])

    def lemma(self, token: str) -> Set[str]:
        return {t.lemma_ for t in self._nlp(token)}


def default_lemmatizer() -> SpacyLemmatizer:
    return SpacyLemmatizer()
