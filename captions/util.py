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

from .index import Lexicon, CaptionIndex

Number = Union[int, float]

VERBOSE = False


def window(tokens: Iterable, n: int, subwindows: bool = False) -> Generator:
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


def frequent_words(
    lexicon: Lexicon, percentile: float = 99.7
) -> List[Lexicon.Word]:
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
    def to_fixed_length(
        postings: Iterable[CaptionIndex.Posting], length: Number,
        duration: Number
    ) -> List[CaptionIndex.Posting]:
        """Dilate start and end times"""
        result = []
        half_length = length / 2
        for p in postings:
            mid = (p.start + p.end) / 2
            result.append(p._replace(
                start=max(mid - half_length, 0),
                end=min(mid + half_length, duration)
            ))
        return result

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

        postings_lists_with_priority = [
            ((get_priority(p), p) for p in pl) for pl in postings_lists]

        return [r[1] for r in heapq.merge(*postings_lists_with_priority)]


def group_results_by_document(
    results: List[Iterable[CaptionIndex.Document]]
) -> Generator[
    Tuple[int, List[List[CaptionIndex.Posting]]], None, None
]:
    """Group postings of documents from multiple results"""
    result_with_id = [((d.id, d) for d in r) for r in results]
    curr_doc_id = None
    curr_docs = []
    for doc_id, document in heapq.merge(*result_with_id):
        if curr_doc_id is None:
            curr_doc_id = doc_id
        if curr_doc_id != doc_id:
            yield curr_doc_id, [d.postings for d in curr_docs]
            curr_doc_id = doc_id
            curr_docs = []
        curr_docs.append(document)
    if len(curr_docs) > 0:
        yield curr_doc_id, [d.postings for d in curr_docs]
