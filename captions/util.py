"""
Higher level search and NLP functionality built over the base index
"""

import heapq
from collections import deque

from .index import InvertedIndex


def window(tokens, n):
    """Takes an iterable words and returns a windowed iterator"""
    assert n > 1, 'Windows of size 1 are silly...'
    buffer = deque()
    for t in tokens:
        buffer.append(t)
        if len(buffer) > n:
            buffer.popleft()
        if len(buffer) == n:
            yield tuple(buffer)


def _dilate_location_results(locations, window):
    for l in locations:
        yield l._replace(
            start=max(l.start - window, 0), end=l.end + window)


def _deoverlap_location_results(location_results):
    new_location_results = deque()
    curr_lr = None
    for lr in location_results:
        if curr_lr is None:
            curr_lr = lr
        elif curr_lr.end >= lr.start:
            curr_lr = curr_lr._replace(
                end=max(lr.end, curr_lr.end),
                min_index=min(lr.min_index, curr_lr.min_index),
                max_index=max(lr.max_index, curr_lr.max_index))
        else:
            new_location_results.append(curr_lr)
            curr_lr = lr
    if curr_lr is not None:
        new_location_results.append(curr_lr)
    return new_location_results


def _dilate_document_results(document_results, window):
    """
    For each document result, add/subtract window to the end/start time of
    each location result
    """
    for dr in document_results:
        location_results = _deoverlap_location_results(
            _dilate_location_results(dr.locations, window))
        yield dr._replace(
            count=len(location_results),
            locations=iter(location_results))


def _union_location_results(location_results):
    """
    Merge several iterators of location_results by order of start
    """
    pq = []
    for i, lr in enumerate(location_results):
        try:
            lr_head = next(lr)
            pq.append((lr_head.start, i, lr_head, lr))
        except StopIteration:
            pass
    heapq.heapify(pq)

    while len(pq) > 0:
        _, i, lr_head, lr = heapq.heappop(pq)
        yield lr_head
        try:
            lr_head = next(lr)
            heapq.heappush(pq, (lr_head.start, i, lr_head, lr))
        except StopIteration:
            pass


def _union_document_results(document_results):
    """
    Merge several iterators of document results and their locations,
    deoverlapping as necessary
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
        curr_doc_loc_results = []
        while len(pq) > 0:
            if pq[0][0] == curr_doc_head.id:
                _, i, dr_head, dr = heapq.heappop(pq)
                curr_doc_loc_results.append(dr_head.locations)
                try:
                    dr_head = next(dr)
                    heapq.heappush(pq, (dr_head.id, i, dr_head, dr))
                except StopIteration:
                    pass
            else:
                break

        if len(curr_doc_loc_results) == 1:
            yield curr_doc_head
        else:
            new_location_results = _deoverlap_location_results(
                _union_location_results(curr_doc_loc_results))
            yield curr_doc_head._replace(
                count=len(new_location_results),
                locations=iter(new_location_results))


def topic_search(phrases, inverted_index, window_size=30):
    """
    Search for time segments where any of the phrases occur with time windows
    dilated by window size seconds.
    """
    assert isinstance(inverted_index, InvertedIndex)
    assert isinstance(phrases, list)
    results = []
    for phrase in phrases:
        try:
            result = inverted_index.search(phrase)
            results.append(result)
        except (KeyError, IndexError, ValueError):
            pass

    document_results = deque()
    for r in results:
        document_results.append(
            _dilate_document_results(r.documents, window_size))
    return InvertedIndex.Result(
        count=None, documents=_union_document_results(document_results))
