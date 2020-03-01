import os
import sys
from subprocess import check_call
from typing import List, NamedTuple

from captions import BinaryFormat

DEFAULT_PARALLELISM = os.cpu_count()

BINARY_FORMAT = BinaryFormat.default()
MAX_WORD_LEN = 20

STDIN_DELIM = '\t'


class DocumentToIndex(NamedTuple):
    name: str
    path: str


def list_docs(dir: str) -> List[DocumentToIndex]:
    return [DocumentToIndex(d, os.path.join(dir, d)) for d in os.listdir(dir)]


def read_docs_from_stdin() -> List[DocumentToIndex]:
    # Read in list of "name path" pairs from stdin
    result = []
    for line in sys.stdin:
        line = line.strip()
        if line != '':
            name, path = [t.strip() for t in line.split(STDIN_DELIM, 1)]
            result.append(DocumentToIndex(name, path))
    return result


def merge_index_files(
    doc_index_paths: List[str], out_path: str,
    batch_size: int = 1000, keep_tmp_files: bool = False
):
    with open(out_path, 'wb') as f:
        for i in range(0, len(doc_index_paths), batch_size):
            max_idx = min(i + batch_size, len(doc_index_paths))
            batch_doc_index_paths = doc_index_paths[i:max_idx]
            check_call(['cat'] + batch_doc_index_paths, stdout=f)
            if not keep_tmp_files:
                for p in batch_doc_index_paths:
                    os.remove(p)
