"""
Pointwise Mutual Information
"""

import math
import sys
import os
from collections import deque, Counter
from multiprocessing import Pool
from typing import Dict, List, Tuple

from .index import Lexicon, CaptionIndex, NgramFrequency
from .util import topic_search, window

VERBOSE = False
DEFAULT_WORKERS = os.cpu_count()

_INDEX = None
_NGRAM_FREQUENCY = None


def _pmi_worker(batch, n):
    length_totals = [0] * n
    ngram_counts = Counter()
    for doc_id, postings in batch:
        for start, end in postings:
            start_pos = _INDEX.position(doc_id, start)
            end_pos = _INDEX.position(doc_id, end)

            # if VERBOSE:
            #     if start_pos <= lr.min_index:
            #         print('Bad start idx: {} > {}'.format(start_pos,
            #               lr.min_index), file=sys.stderr)
            #     if end_pos >= lr.max_index:
            #         print('Bad end idx: {} < {}'.format(end_pos, lr.max_index),
            #               file=sys.stderr)

            for ngram in window(_INDEX.tokens(doc_id, start_pos,
                                end_pos), n, True):
                if ngram in _NGRAM_FREQUENCY:
                    ngram_counts[ngram] += 1
                length_totals[len(ngram) - 1] += 1
    return length_totals, ngram_counts


def compute(phrases: List, index: CaptionIndex,
            ngram_frequency: NgramFrequency, n: int,
            window_size=30, workers=DEFAULT_WORKERS,
            batch_size=10000) -> Dict[Tuple, float]:
    """
    Get ngrams that occur in the context of the query phrases.
    """
    intervals = topic_search(phrases, index, window_size)

    length_totals = [0] * n
    ngram_scores = Counter()

    global _INDEX, _NGRAM_FREQUENCY
    _INDEX = index
    _NGRAM_FREQUENCY = ngram_frequency
    with Pool(processes=workers) as pool:
        batch = deque()
        results = deque()
        for dr in intervals:
            postings = deque()
            for p in dr.postings:
                postings.append((p.start, p.end))
            batch.append((dr.id, postings))

            if len(batch) == batch_size:
                results.append(pool.apply_async(_pmi_worker, (batch, n)))
                batch = deque()

        if len(batch) > 0:
            results.append(pool.apply_async(_pmi_worker, (batch, n)))

        for result in results:
            batch_length_totals, batch_ngram_counts = result.get()
            for i, v in enumerate(batch_length_totals):
                length_totals[i] += v
            for k in batch_ngram_counts:
                ngram_scores[k] += batch_ngram_counts[k]

    log_length_totals = [math.log(x) if x > 0 else float('-inf')
                         for x in length_totals]
    for k, v in ngram_scores.items():
        ngram_scores[k] = (math.log(v) - log_length_totals[len(k) - 1] -
                           math.log(ngram_frequency[k]))
    return ngram_scores
