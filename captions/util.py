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


def _deoverlap_postings(
    postings: Iterable[CaptionIndex.Posting]
) -> Iterable[CaptionIndex.Posting]:
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


def topic_search(phrases: List, index: CaptionIndex, window_size=30,
                 documents=None):
    """
    Search for time segments where any of the phrases occur with time windows
    dilated by window size seconds.
    """
    assert isinstance(index, CaptionIndex)
    assert isinstance(phrases, list)
    results = []
    for phrase in phrases:
        try:
            result = index.search(phrase, documents=documents)
            results.append(_dilate_search_results(index, result, window_size))
        except (KeyError, IndexError, ValueError):
            pass
    return _union_search_results(results)
