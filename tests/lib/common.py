import os
import captions


def get_docs_and_lexicon(idx_dir):
    doc_path = os.path.join(idx_dir, 'documents.txt')
    lex_path = os.path.join(idx_dir, 'lexicon.txt')
    data_path = os.path.join(idx_dir, 'data')

    documents = captions.Documents.load(doc_path)
    documents.configure(data_path)
    lexicon = captions.Lexicon.load(lex_path)
    return documents, lexicon
