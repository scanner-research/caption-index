from collections import Counter
from multiprocessing import Pool
import os
import sys
from subprocess import check_call
from typing import List, NamedTuple

from tqdm import tqdm

from captions import BinaryFormat
from captions.indexer import get_document_word_counts

DEFAULT_PARALLELISM = os.cpu_count()

BINARY_FORMAT = BinaryFormat.default()
MAX_WORD_LEN = 20

STDIN_DELIM = '\t'


class DocumentToIndex(NamedTuple):
    name: str
    path: str


def list_docs(doc_dir: str) -> List[DocumentToIndex]:
    return [DocumentToIndex(d, os.path.join(doc_dir, d))
            for d in os.listdir(doc_dir)]


def read_docs_from_stdin() -> List[DocumentToIndex]:
    # Read in list of "name path" pairs from stdin
    result = []
    for line in sys.stdin:
        line = line.strip()
        if line != '':
            name, path = [t.strip() for t in line.split(STDIN_DELIM, 1)]
            result.append(DocumentToIndex(name, path))
    return result


def merge_files(
        paths: List[str], out_path: str,
        batch_size: int = 1000, keep_tmp_files: bool = False
):
    with open(out_path, 'wb') as f:
        for i in range(0, len(paths), batch_size):
            max_idx = min(i + batch_size, len(paths))
            batch_paths = paths[i:max_idx]
            check_call(['cat'] + batch_paths, stdout=f)
            if not keep_tmp_files:
                for p in batch_paths:
                    os.remove(p)


def get_doc_word_counts(doc_path: str) -> Counter:
    return get_document_word_counts(doc_path, max_word_len=MAX_WORD_LEN)


def get_word_counts(docs_to_index: List[DocumentToIndex], parallelism: int):
    words = Counter()
    with Pool(processes=parallelism) as pool:
        for result in tqdm(pool.imap_unordered(
                get_doc_word_counts, [d.path for d in docs_to_index]
        ), desc='Building lexicon', total=len(docs_to_index)):
            words += result

    print('Lexicon size: {}'.format(len(words)))
    return words
