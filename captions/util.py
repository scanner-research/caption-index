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
from typing import Generator, Iterable, List, Tuple, Union

from .index import Lexicon, CaptionIndex, NgramFrequency

Number = Union[int, float]

VERBOSE = False


def window(tokens: Iterable, n: int, subwindows: bool = False):
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


def frequent_words(lexicon: Lexicon, percentile: float = 99.7) -> List[Lexicon.Word]:
    """Return words at a frequency percentile"""
    threshold = np.percentile([w.count for w in lexicon], percentile)
    return [w for w in lexicon if w.count >= threshold]


class PostingUtil(object):

    @staticmethod
    def merge(p1: CaptionIndex.Posting, p2: CaptionIndex.Posting):
        """Merge two postings"""
        start_idx = min(p1.idx, p2.idx)
        end_idx = max(p1.idx + p1.len, p2.idx + p2.len)
        return p1._replace(
            start=min(p1.start, p2.start),
            end=max(p1.end, p2.end),
            idx=start_idx,
            len=end_idx - start_idx)

    @staticmethod
    def deoverlap(
        postings: Iterable[CaptionIndex.Posting], threshold: Number = 0,
        use_time: bool = True
    ) -> List[CaptionIndex.Posting]:
        """Merge postings which overlap"""
        result = []
        curr_p = None

        def overlaps(p1: CaptionIndex.Posting,
                     p2: CaptionIndex.Posting) -> bool:
            if use_time:
                return (p2.start >= p1.start
                        and p2.start - p1.end <= threshold)
            else:
                return (p2.idx >= p1.idx
                        and p2.idx - (p1.idx + p1.len) <= threshold)

        for p in postings:
            if curr_p is None:
                curr_p = p
            elif overlaps(curr_p, p):
                curr_p = PostingUtil.merge(curr_p, p)
            else:
                result.append(curr_p)
                curr_p = p
        if curr_p is not None:
            result.append(curr_p)
        return result

    @staticmethod
    def dilate(
        postings: Iterable[CaptionIndex.Posting], window: Number,
        duration: Number
    ) -> List[CaptionIndex.Posting]:
        """Dilate start and end times"""
        return [p._replace(
                    start=max(p.start - window, 0),
                    end=min(p.end + window, duration)
                ) for p in postings]

    @staticmethod
    def union(
        postings_lists: List[Iterable[CaptionIndex.Posting]],
        use_time: bool = True
    ) -> List[CaptionIndex.Posting]:
        """Merge several lists of postings by order of idx."""
        def get_priority(p: CaptionIndex.Posting) -> Tuple[Number, Number]:
            if use_time:
                return (p.start, p.idx)
            else:
                return (p.idx, p.start)

        result = []
        pq = []
        for i, ps_list in enumerate(postings_lists):
            ps_iter = iter(ps_list)
            ps_head = next(ps_iter)
            pq.append((get_priority(ps_head), i, ps_head, ps_iter))
        heapq.heapify(pq)

        while len(pq) > 0:
            _, i, ps_head, ps_iter = heapq.heappop(pq)
            result.append(ps_head)
            try:
                ps_head = next(ps_iter)
                heapq.heappush(pq, (get_priority(ps_head), i, ps_head,
                                    ps_iter))
            except StopIteration:
                pass
        return result


def group_results_by_document(
    results: List[Iterable[CaptionIndex.Document]]
) -> Generator[
    Tuple[int, List[List[CaptionIndex.Posting]]], None, None
]:
    """Group postings of documents from multiple results"""
    pq = []
    for i, docs in enumerate(results):
        try:
            doc_head = next(docs)
            pq.append((doc_head.id, i, doc_head, docs))
        except StopIteration:
            pass
    heapq.heapify(pq)

    while len(pq) > 0:
        curr_doc_head = pq[0][2]
        curr_doc_postings_lists = []
        while len(pq) > 0:
            if pq[0][0] == curr_doc_head.id:
                _, i, doc_head, docs = heapq.heappop(pq)
                curr_doc_postings_lists.append(doc_head.postings)
                try:
                    doc_head = next(docs)
                    heapq.heappush(pq, (doc_head.id, i, doc_head, docs))
                except StopIteration:
                    pass
            else:
                break
        yield curr_doc_head.id, curr_doc_postings_lists


def topic_search(
    phrases: List, index: CaptionIndex, window_size: Number = 30,
    documents=None
) -> Iterable[CaptionIndex.Document]:
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
            results.append(result)
        except (KeyError, IndexError, ValueError):
            pass
    for doc_id, posting_lists in group_results_by_document(results):
        duration = index.document_duration(doc_id)
        postings = PostingUtil.deoverlap(
            PostingUtil.dilate(
                PostingUtil.union(posting_lists),
                window_size, duration))
        yield CaptionIndex.Document(id=doc_id, postings=postings)
