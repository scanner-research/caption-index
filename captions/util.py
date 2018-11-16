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

from .index import Lexicon, InvertedIndex, DocumentData, NgramFrequency


VERBOSE = False


def window(tokens, n, subwindows=False):
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


def frequent_words(lexicon, percentile=99.7):
    assert isinstance(lexicon, Lexicon)
    threshold = np.percentile([w.count for w in lexicon], percentile)
    return [w for w in lexicon if w.count >= threshold]


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
            if VERBOSE and curr_lr.start <= lr.start:
                print('{} > {}'.format(curr_lr.start, lr.start),
                      file=sys.stderr)
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


_DOCUMENT_DATA = None
_NGRAM_FREQUENCY = None


def _pmi_worker(batch, n):
    length_totals = [0] * n
    ngram_counts = Counter()
    for doc_id, locs in batch:
        for start, end in locs:
            start_pos = _DOCUMENT_DATA.time_to_position(doc_id, start)
            end_pos = _DOCUMENT_DATA.time_to_position(doc_id, end)

            # if VERBOSE:
            #     if start_pos <= lr.min_index:
            #         print('Bad start idx: {} > {}'.format(start_pos,
            #               lr.min_index), file=sys.stderr)
            #     if end_pos >= lr.max_index:
            #         print('Bad end idx: {} < {}'.format(end_pos, lr.max_index),
            #               file=sys.stderr)

            for ngram in window(_DOCUMENT_DATA.tokens(doc_id, start_pos,
                                end_pos), n, True):
                if ngram in _NGRAM_FREQUENCY:
                    ngram_counts[ngram] += 1
                length_totals[len(ngram) - 1] += 1
    return length_totals, ngram_counts


DEFAULT_WORKERS = os.cpu_count()


def pmi_search(phrases, inverted_index, document_data, ngram_frequency,
               n, window_size=30, workers=DEFAULT_WORKERS, batch_size=10000):
    """
    Get ngrams that occur in the context of the query phrases.
    """
    assert isinstance(inverted_index, InvertedIndex)
    assert isinstance(document_data, DocumentData)
    assert isinstance(ngram_frequency, NgramFrequency)
    intervals = topic_search(phrases, inverted_index, window_size)

    length_totals = [0] * n
    ngram_scores = Counter()

    global _DOCUMENT_DATA, _NGRAM_FREQUENCY
    _DOCUMENT_DATA = document_data
    _NGRAM_FREQUENCY = ngram_frequency
    with Pool(processes=workers) as pool:
        batch = deque()
        results = deque()
        for dr in intervals.documents:
            locations = deque()
            for lr in dr.locations:
                locations.append((lr.start, lr.end))
            batch.append((dr.id, locations))

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
