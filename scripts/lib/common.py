import os
from subprocess import check_call
from typing import List

from captions import BinaryFormat

DEFAULT_PARALLELISM = os.cpu_count()
DEFAULT_SOURCE_FILE_EXT = 'srt'

BINARY_FORMAT = BinaryFormat.default()
MAX_WORD_LEN = 20


def list_docs(dir: str, ext: str) -> List[str]:
    return [f for f in os.listdir(dir) if f.endswith(ext)]


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
