"""
Text search query language.

[ Grammar ]

Phrases:
  the ducks        find all instances of the "the ducks"

Query Expansion: [] brackets
  [the ducks]     "the ducks" or "the duck"

Or: |
  e1 | e2 | ...   expression1 or expression2

And: &
  e1 & e2 & ...   expression1 and expression2 nearby
  e1 & e2 :: t    same as above, but with t seconds as the threshold

Not: &
  e1 ^ e2 ^ ...   expression1, not near expression2
  e1 ^ e2 :: t    same as above, but with t seconds as the threshold

Groups: ()
  (expr)          evaluate expr as a group

Caveats
    &, |, and ^ cannot be combined in the same group. For instance,
    "(a & b | c)" is invalid and should be written as "(a & (b | c))" or
    "((a & b) | c)".

[ Query Examples ]

  united states
    All instances of the "united states".

  united | states
    All instances of "united" or "states".

  united & states
    All instances of "united" and "states" nearby.

  united ^ states
    All instances of "united", without "states" nearby.

  united ^ (states | kingdom)
    All instances of "united", without "states" or "kingdom" nearby.

"""

import heapq
from abc import ABC, abstractmethod, abstractproperty
from collections import deque, namedtuple
from typing import Dict, List, Iterable
from parsimonious.grammar import Grammar, NodeVisitor

from .index import Lexicon, CaptionIndex


GRAMMAR = Grammar(r"""
    expr_group = and / or / not / expr
    expr = expr_paren / tokens_root
    expr_paren = "(" sp? expr_group sp? ")"

    and = expr more_and threshold?
    more_and = (sp? "&" sp? expr)+

    or = expr more_or
    more_or = (sp? "|" sp? expr)+

    not = expr more_not threshold?
    more_not = (sp? "^" sp?  expr)+

    threshold = sp? "::" sp? integer
    integer = ~r"\d+"

    tokens_root = tokens_list more_tokens_root
    more_tokens_root = (sp tokens_list)*
    tokens_list = tokens / tokens_exp

    tokens_exp = "[" sp? tokens sp? "]"
    tokens = token more_tokens
    more_tokens = (sp token)*

    token = ~r"[\w\-\_\$\!]+"

    sp = ~r"\s+"
""")


def _group_by_document(results: List[Iterable['CaptionIndex.Document']]):
    """Group postings of documents."""
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


def _union_postings(postings):
    """Merge several lists of postings by order of idx."""
    result = []
    pq = []
    for i, ps_list in enumerate(postings):
        ps_iter = iter(ps_list)
        ps_head = next(ps_iter)
        pq.append((ps_head.start, i, ps_head, ps_iter))
    heapq.heapify(pq)

    while len(pq) > 0:
        _, i, ps_head, ps_iter = heapq.heappop(pq)
        result.append(ps_head)
        try:
            ps_head = next(ps_iter)
            heapq.heappush(pq, (ps_head.start, i, ps_head, ps_iter))
        except StopIteration:
            pass
    return result


class _Expr(ABC):

    Context = namedtuple('Context', ['lexicon', 'index', 'documents'])

    @abstractmethod
    def eval(self, context) -> Iterable['CaptionIndex.Document']:
        raise NotImplementedError()

    @abstractproperty
    def _pprint_data(self):
        raise NotImplementedError()

    def __repr__(self):
        return repr(self._pprint_data)


class _JoinExpr(_Expr):

    def __init__(self, children, threshold):
        assert all(isinstance(c, _Expr) for c in children)
        self.children = children
        self.threshold = threshold


class _Phrase(_Expr):

    Token = namedtuple('Token', ['text', 'expand'])

    def __init__(self, tokens):
        assert all(isinstance(t, _Phrase.Token) for t in tokens)
        self.tokens = tokens

    @property
    def _pprint_data(self):
        return {
            '1. op': 'Phrase',
            '2. tokens': ' '.join([
                '[{}]'.format(t.text) if t.expand else t.text
                for t in self.tokens])
        }

    def eval(self, context):
        kwargs = {}
        if context.documents is not None:
            kwargs['documents'] = context.documents
        results = []

        def helper(i, tokens):
            if i == len(self.tokens):
                results.append(context.index.ngram_search(*tokens, **kwargs))
                return
            t = self.tokens[i]
            if t.expand:
                for t_ in context.lexicon.similar(t.text):
                    tokens.append(context.lexicon[t_])
                    helper(i + 1, tokens)
                    tokens.pop()
            else:
                w = context.lexicon[t.text]
                tokens.append(w)
                helper(i + 1, tokens)
                tokens.pop()
        helper(0, deque())

        for doc_id, grouped_postings in _group_by_document(results):
            yield CaptionIndex.Document(
                id=doc_id, postings=_union_postings(grouped_postings))


def _dist_posting(p1, p2):
    return (
        max(p2.start - p1.end, 0)
        if p1.start <= p2.start else _dist_posting(p2, p1))


class _And(_JoinExpr):

    @property
    def _pprint_data(self):
        return {
            '1. op': 'And',
            '2. thresh': '{} seconds'.format(self.threshold),
            '3. children': [c._pprint_data for c in self.children]
        }

    def eval(self, context):
        results = []
        for c in self.children:
            child_results = deque(c.eval(context))
            doc_ids = [d.id for d in child_results]
            context = context._replace(documents=doc_ids)
            results.append({d.id: d.postings for d in child_results})

        n = len(results)
        for doc_id in sorted(doc_ids):
            pq = []
            i = 0
            for r in results:
                ps_iter = iter(r[doc_id])
                ps_head = next(ps_iter)
                pq.append((ps_head.start, i, ps_head, ps_iter))
                i += 1
            heapq.heapify(pq)

            merged_postings = []
            ps_prev = [None] * n
            while len(pq) > 0:
                # Consider first element
                _, i, ps_head, ps_iter = heapq.heappop(pq)

                # Check conditions
                near_i = set()
                for elem in pq:
                    ps_cmp = elem[2]
                    j = elem[1]
                    if _dist_posting(ps_head, ps_cmp) < self.threshold:
                        near_i.add(j)
                if len(near_i) < n - 1:
                    for j in range(n):
                        if j != i and j not in near_i:
                            ps_cmp = ps_prev[j]
                            if ps_cmp is not None:
                                if _dist_posting(ps_head, ps_cmp) < self.threshold:
                                    near_i.add(j)
                            else:
                                # No solution
                                break
                if len(near_i) == n - 1:
                    merged_postings.append(ps_head)

                # Advance postings
                ps_prev[i] = ps_head
                try:
                    ps_head = next(ps_iter)
                    heapq.heappush(pq, (ps_head.start, i, ps_head, ps_iter))
                    i += 1
                except StopIteration:
                    pass

            merged_postings.sort(key=lambda x: x.start)
            if len(merged_postings) > 0:
                yield CaptionIndex.Document(
                    id=doc_id, postings=merged_postings)


class _Or(_JoinExpr):

    @property
    def _pprint_data(self):
        return {
            '1. op': 'Or',
            '2. children': [c._pprint_data for c in self.children]
        }

    def eval(self, context):
        results = [c.eval(context) for c in self.children]
        for doc_id, grouped_postings in _group_by_document(results):
            yield CaptionIndex.Document(
                id=doc_id,
                postings=_union_postings(grouped_postings)
            )


class _Not(_JoinExpr):

    @property
    def _pprint_data(self):
        return {
            '1. op': 'Not',
            '2. thresh': '{} seconds'.format(self.threshold),
            '3. children': [c._pprint_data for c in self.children]
        }

    def eval(self, context):
        child0_results = list(self.children[0].eval(context))

        other_context = context._replace(
            documents=[d.id for d in child0_results])
        other_results = [c.eval(other_context) for c in self.children[1:]]
        other_postings = {
            doc_id: _union_postings(grouped_postings)
            for doc_id, grouped_postings in _group_by_document(other_results)
        }

        # TODO: this is a silly way to join
        for d in child0_results:
            postings = []
            for p1 in d.postings:
                p1_start_t = p1.start - self.threshold
                p1_end_t = p1.end + self.threshold
                for p2 in other_postings.get(d.id, []):
                    if min(p1_end_t, p2.end) - max(p1_start_t, p2.start) >= 0:
                        break
                else:
                    postings.append(p1)
            if len(postings) > 0:
                yield CaptionIndex.Document(id=d.id, postings=postings)


DEFAULT_AND_THRESH = 60
DEFAULT_NOT_THRESH = 60


class QueryParser(NodeVisitor):

    def __init__(self, constants={}):
        self.grammar = GRAMMAR
        self._constants = constants

    visit_more_and = visit_more_or = visit_more_not = visit_more_tokens = \
        lambda a, b, c: c

    def visit_expr_group(self, node, children):
        assert len(children) == 1
        return children[0]

    def visit_expr(self, node, children):
        assert len(children) == 1
        return children[0]

    def visit_expr_paren(self, node, children):
        assert len(children) == 5
        return children[2]

    def visit_and(self, node, children):
        assert len(children) == 3
        if children[2] is None:
            threshold = self._constants.get('and_threshold', DEFAULT_AND_THRESH)
        else:
            threshold = children[2]
            assert isinstance(threshold, int)
        return _And([children[0], *children[1]], threshold)

    def visit_or(self, node, children):
        assert len(children) == 2
        return _Or([children[0], *children[1]], None)

    def visit_not(self, node, children):
        assert len(children) == 3
        if children[2] is None:
            threshold = self._constants.get('not_threshold', DEFAULT_NOT_THRESH)
        else:
            threshold = children[2]
            assert isinstance(threshold, int)
        return _Not([children[0], *children[1]], threshold)

    def visit_threshold(self, node, children):
        assert len(children) == 4
        return children[3]

    def visit_integer(self, node, children):
        return int(node.text)

    def visit_tokens_root(self, node, children):
        assert len(children) == 2
        return _Phrase([*children[0], *children[1]])

    def visit_more_tokens_root(self, node, children):
        return [l for c in children for l in c]

    def visit_tokens_list(self, node, children):
        assert len(children) == 1
        return children[0]

    def visit_tokens_exp(self, node, children):
        assert len(children) == 5
        return [t._replace(expand=True) for t in children[2]]

    def visit_tokens(self, node, children):
        assert len(children) == 2
        return [children[0], *children[1]]

    def visit_token(self, node, children):
        return _Phrase.Token(node.text, False)

    def generic_visit(self, node, children):
        return children[-1] if children else None


class Query(object):

    def __init__(self, raw_query: str, **config):
        self._tree = QueryParser(config).parse(raw_query)

    def execute(self, lexicon: Lexicon, index: CaptionIndex, documents=None):
        return self._tree.eval(_Expr.Context(lexicon, index, documents))
