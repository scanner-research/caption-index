"""
Higher level search and NLP functionality built over the base index
"""

import heapq
import itertools
import math
import numpy as np
import sys
import os
from collections import deque, Counter
from multiprocessing import Pool
from typing import Iterable, List

from .index import Lexicon, CaptionIndex, NgramFrequency


VERBOSE = False


def window(tokens: Iterable, n: int, subwindows=False):
    """Takes an iterable words and returns a windowed iterator"""
    buffer = deque()
    for t in tokens:
        buffer.append(t)
        if len(buffer) == n:
            if subwindows:
                for i in range(n):
                    yield tuple(itertools.islice(buffer, 0, i + 1))
            else:
                yield tuple(buffer)
            buffer.popleft()
    if subwindows:
        while len(buffer) > 0:
            for i in range(len(buffer)):
                yield tuple(itertools.islice(buffer, 0, i + 1))
            buffer.popleft()


def frequent_words(lexicon: Lexicon, percentile=99.7) -> List[Lexicon.Word]:
    threshold = np.percentile([w.count for w in lexicon], percentile)
    return [w for w in lexicon if w.count >= threshold]


def _dilate_postings(postings: Iterable[CaptionIndex.Posting], window: int,
                     duration: float) -> Iterable[CaptionIndex.Posting]:
    return [p._replace(
        start=max(p.start - window, 0), end=min(p.end + window, duration)
    ) for p in postings]


def _deoverlap_postings(postings: Iterable[CaptionIndex.Posting]) -> Iterable[CaptionIndex.Posting]:
    new_postings = deque()
    curr_posting = None
    for p in postings:
        if curr_posting is None:
            curr_posting = p
        elif curr_posting.end >= p.start:
            if VERBOSE and curr_posting.start <= p.start:
                print('{} > {}'.format(curr_posting.start, p.start),
                      file=sys.stderr)
            min_idx = min(p.idx, curr_posting.idx)
            max_idx = max(p.idx + p.len, curr_posting.idx + curr_posting.len)
            curr_posting = curr_posting._replace(
                end=max(p.end, curr_posting.end), idx=min_idx,
                len=max_idx - min_idx)
        else:
            new_postings.append(curr_posting)
            curr_posting = p
    if curr_posting is not None:
        new_postings.append(curr_posting)
    return new_postings


def _dilate_search_results(index: CaptionIndex,
                           results: Iterable[CaptionIndex.Document],
                           window: int) -> Iterable[CaptionIndex.Document]:
    """
    For each document's result, add/subtract window to the end/start time of
    each posting.
    """
    for d in results:
        duration = index.document_duration(d.id)
        postings = _deoverlap_postings(
            _dilate_postings(d.postings, window, duration))
        yield d._replace(postings=postings)


def _union_postings(postings_lists):
    """
    Merge several lists of postings by order of start time.
    """
    result = []
    pq = []
    for i, ps in enumerate(postings_lists):
        assert isinstance(ps, deque)
        ps_head = ps.popleft()
        pq.append((ps_head.start, i, ps_head, ps))
    heapq.heapify(pq)

    while len(pq) > 0:
        _, i, ps_head, ps = heapq.heappop(pq)
        result.append(ps_head)
        if len(ps) > 0:
            ps_head = ps.popleft()
            heapq.heappush(pq, (ps_head.start, i, ps_head, ps))
    return result


def _union_search_results(document_results):
    """
    Merge several iterators of document results and their postings,
    deoverlapping as necessary.
    """
    pq = []
    for i, dr in enumerate(document_results):
        try:
            dr_head = next(dr)
            pq.append((dr_head.id, i, dr_head, dr))
        except StopIteration:
            pass
    heapq.heapify(pq)

    while len(pq) > 0:
        curr_doc_head = pq[0][2]
        curr_doc_postings_lists = []
        while len(pq) > 0:
            if pq[0][0] == curr_doc_head.id:
                _, i, dr_head, dr = heapq.heappop(pq)
                curr_doc_postings_lists.append(dr_head.postings)
                try:
                    dr_head = next(dr)
                    heapq.heappush(pq, (dr_head.id, i, dr_head, dr))
                except StopIteration:
                    pass
            else:
                break

        if len(curr_doc_postings_lists) == 1:
            yield curr_doc_head
        else:
            new_postings = _deoverlap_postings(
                _union_postings(curr_doc_postings_lists))
            yield curr_doc_head._replace(postings=new_postings)


def topic_search(phrases: List, index: CaptionIndex, window_size=30):
    """
    Search for time segments where any of the phrases occur with time windows
    dilated by window size seconds.
    """
    assert isinstance(index, CaptionIndex)
    assert isinstance(phrases, list)
    results = []
    for phrase in phrases:
        try:
            result = index.search(phrase)
            results.append(_dilate_search_results(index, result, window_size))
        except (KeyError, IndexError, ValueError):
            pass
    return _union_search_results(results)


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


DEFAULT_WORKERS = os.cpu_count()


def pmi_search(phrases: List, index: CaptionIndex,
               ngram_frequency: NgramFrequency, n: int,
               window_size=30, workers=DEFAULT_WORKERS, batch_size=10000):
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
