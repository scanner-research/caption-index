import os
import sys
from subprocess import check_call
from typing import List, Dict, NamedTuple, Optional

from captions import BinaryFormat, Lexicon, Documents
from captions.rs_captions import indexer

BINARY_FORMAT = BinaryFormat()
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

def get_word_counts(
        docs_to_index: List[DocumentToIndex],
        batch_size: Optional[int] = None
) -> Dict[str, int]:
    if batch_size is None:
        batch_size = int(len(docs_to_index) / 10 / os.cpu_count())
        batch_size = min(max(batch_size, 1), 1000)
    assert batch_size > 0
    doc_paths = [d.path for d in docs_to_index]
    words = indexer.count_tokens(doc_paths, MAX_WORD_LEN, batch_size, True)
    print('Lexicon size: {}'.format(len(words)))
    return words


def index_documents(
        index_and_doc_paths, lexicon: Lexicon,
        binary_format: BinaryFormat = BINARY_FORMAT
):
    indexer.index_documents(
        index_and_doc_paths, {w.token: w.id for w in lexicon}, True,
        binary_format.datum_bytes, binary_format.start_time_bytes,
        binary_format.end_time_bytes)
