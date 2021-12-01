"""
Text search query language.

[ Grammar ]

Phrases:
  the ducks             find all instances of the "the ducks" (a bigram)

Query Expansion: [] brackets
  [the ducks]           "the ducks" or "the duck".
                        Equivalent to ("the duck" | "the ducks").

Expressions:
  An expression can be:
    - a phrase  (e.g., "the ducks")
    - a phrase with query expansion (e.g., [the ducks])
    - composition of expressions with the operators below
  The result of evaluating an expression is always a list of locations where
  phrases that match the expression appear. There is no merging of phrases
  in the text query language.

Or: |
  e1 | e2 | e3 | ...    expr1 or expr2 or expr3

And: &
  e1 & e2 & e3 & ...    expr1, expr2, and expr3 where they are nearby
                        I.e., instances of e1, e2, e3 that are contained in
                        a window that contains an instance of e1, e2, and e3

  e1 & e2 & e3 :: t     same as above, but with t seconds as the window
                        threshold

  e1 & e2 & e3 // w     same as above, but with w tokens as the threshold

Not near: \
  e1 \ e2 \ e3 ...      expr1, not near expr2 and not near expr3
                        I.e., instances of e1 not in any window containing
                        e1 or e2.

                        This is equivalent to (e1 \ (e2 | e3)) and
                        ((e1 \ e2) \ e3).

  e1 \ e2 \ e3 :: t     same as above, but with t seconds as the window

  e1 \ e2 \ e3 // w     same as above, but with w tokens as the threshold

Groups: ()
  (expr)          evaluate expr as a group

[ Group caveats ]
    &, |, and \ cannot be combined in the same group. For instance,
    "(a & b | c)" is invalid and should be written as "(a & (b | c))" or
    "((a & b) | c)".

[ Query Examples ]

  united states
    All instances of the "united states".

  united | states
    All instances of "united" and "states".

  united & states
    All instances of "united" and "states" where each "united" is near a
    "states" and each "states" is near a "united".

  united \ states
    All instances of "united", without "states" nearby.

  united \ (states | kingdom) ==equiv== united \ states \ kingdom)
    All instances of "united", without "states" and without "kingdom" nearby.
"""

import heapq
from abc import ABC, abstractmethod, abstractproperty
from collections import deque
from typing import Dict, List, Iterable, NamedTuple, Optional
from parsimonious.grammar import Grammar, NodeVisitor

from .index import Lexicon, CaptionIndex
from .tokenize import default_tokenizer
from .util import PostingUtil, group_results_by_document


GRAMMAR = Grammar(r"""
    expr_root = sp? expr_group sp?

    expr_group = and / or / not / expr
    expr = expr_paren / tokens_root
    expr_paren = sp? "(" sp? expr_group sp? ")" sp?

    and = expr more_and threshold?
    more_and = (sp? "&" sp? expr)+

    or = expr more_or
    more_or = (sp? "|" sp? expr)+

    not = expr more_not threshold?
    more_not = (sp? "\\" sp?  expr)+

    threshold = sp? threshold_type sp? integer
    threshold_type = "::" / "//"
    integer = ~r"\d+"

    tokens_root = tokens_list more_tokens_root
    more_tokens_root = (sp tokens_list)*
    tokens_list = tokens / tokens_exp

    tokens_exp = "[" sp? tokens sp? "]"
    tokens = token more_tokens
    more_tokens = (sp token)*

    token = ~r"[^\s()&|\\\[\]:/]+"

    sp = ~r"\s+"
""")


class _Expr(ABC):

    class Context(NamedTuple):
        lexicon: Lexicon
        index: CaptionIndex
        documents: Optional[Iterable[CaptionIndex.DocIdOrDocument]]
        ignore_word_not_found: bool
        case_insensitive: bool

    @abstractmethod
    def eval(self, context: '_Expr.Context') -> Iterable[CaptionIndex.Document]:
        raise NotImplementedError()

    @abstractmethod
    def estimate_cost(self, lexicon: Lexicon) -> float:
        raise NotImplementedError()

    @abstractproperty
    def _pprint_data(self):
        raise NotImplementedError()

    def __repr__(self):
        return repr(self._pprint_data)


class _JoinExpr(_Expr):

    def __init__(self, children, threshold, threshold_type):
        assert all(isinstance(c, _Expr) for c in children)
        self.children = children
        self.threshold = threshold
        self.threshold_type = threshold_type

    def estimate_cost(self, lexicon):
        return sum(c.estimate_cost(lexicon) for c in self.children)


class _Phrase(_Expr):

    class Token(NamedTuple):
        text: str
        expand: bool

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

        ngram_tokens = []
        for t in self.tokens:
            if t.expand:
                tokens = [context.lexicon[x] for x in
                          context.lexicon.similar(t.text)]
                if len(tokens) == 0:
                    return
                ngram_tokens.append(tokens)
            else:
                try:
                    tokens = [context.lexicon[t.text]]
                    if context.case_insensitive:
                        matches = context.lexicon.case_insensitive(tokens[0])
                        if matches is not None:
                            # Some other words exist
                            assert len(matches) > 0
                            tokens = [context.lexicon[x] for x in matches]
                except Lexicon.WordDoesNotExist:
                    if context.ignore_word_not_found:
                        return
                    else:
                        raise
                ngram_tokens.append(tokens)

        for d in context.index.ngram_search(*ngram_tokens, **kwargs):
            yield d

    def estimate_cost(self, lexicon):
        # The cost to search is the frequency of the least frequent token
        # in the ngram since this is the number of locations that need to
        # be checked.
        min_token_count = lexicon.word_count
        for t in self.tokens:
            token_count = 0
            if t.expand:
                tokens = [lexicon[x] for x in
                          lexicon.similar(t.text)]
                token_count += sum(x.count for x in tokens)
            else:
                # FIXME: Case insensitivity not considered here
                try:
                    token = lexicon[t.text]
                    token_count += token.count
                except Lexicon.WordDoesNotExist:
                    pass
            min_token_count = min(token_count, min_token_count)
        return min_token_count / lexicon.word_count


def _dist_time_posting(p1, p2):
    return (
        max(p2.start - p1.end, 0)
        if p1.start <= p2.start else _dist_time_posting(p2, p1))


def _dist_idx_posting(p1, p2):
    return (
        max(p2.idx - (p1.idx + p1.len), 0)
        if p1.idx <= p2.idx else _dist_idx_posting(p2, p1))


class _And(_JoinExpr):

    @property
    def _pprint_data(self):
        return {
            '1. op': 'And',
            '2. thresh': '{} {}'.format(
                self.threshold,
                'seconds' if self.threshold_type == 't' else 'tokens'),
            '3. children': [c._pprint_data for c in self.children]
        }

    def eval(self, context):
        results = []
        for c in self.children:
            child_results = deque(c.eval(context))
            if len(child_results) == 0:
                return

            doc_ids = [d.id for d in child_results]
            context = context._replace(documents=doc_ids)
            results.append({d.id: d.postings for d in child_results})

        dist_fn = (
            _dist_time_posting if self.threshold_type == 't' else
            _dist_idx_posting)

        n = len(results)
        for doc_id in sorted(doc_ids):
            pq = []
            for i, r in enumerate(results):
                assert doc_id in r
                ps_iter = iter(r[doc_id])
                ps_head = next(ps_iter)
                pq.append((ps_head.start, i, ps_head, ps_iter))
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
                    if dist_fn(ps_head, ps_cmp) < self.threshold:
                        near_i.add(j)
                if len(near_i) < n - 1:
                    for j in range(n):
                        if j != i and j not in near_i:
                            ps_cmp = ps_prev[j]
                            if ps_cmp is not None:
                                if dist_fn(ps_head, ps_cmp) < self.threshold:
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
        for doc_id, grouped_postings in group_results_by_document(results):
            yield CaptionIndex.Document(
                id=doc_id,
                postings=PostingUtil.union(grouped_postings)
            )


class _Not(_JoinExpr):

    @property
    def _pprint_data(self):
        return {
            '1. op': 'Not',
            '2. thresh': '{} {}'.format(
                self.threshold,
                'seconds' if self.threshold_type == 't' else 'tokens'),
            '3. children': [c._pprint_data for c in self.children]
        }

    def eval(self, context):
        child0_results = list(self.children[0].eval(context))

        other_context = context._replace(
            documents=[d.id for d in child0_results])
        other_results = [c.eval(other_context) for c in self.children[1:]]
        other_postings = {
            doc_id: PostingUtil.union(ps_lists)
            for doc_id, ps_lists in group_results_by_document(other_results)
        }

        dist_fn = (
            _dist_time_posting if self.threshold_type == 't' else
            _dist_idx_posting)

        key_fn = (
            (lambda x: x.start) if self.threshold_type == 't' else
            (lambda x: x.idx))

        for d in child0_results:
            postings = []
            doc_ops = other_postings.get(d.id, [])
            doc_op_i = 0

            prev_op = None
            for p in d.postings:
                p_key = key_fn(p)
                while (
                    doc_op_i < len(doc_ops)
                    and key_fn(doc_ops[doc_op_i]) <= p_key
                ):
                    prev_op = doc_ops[doc_op_i]
                    doc_op_i += 1

                if prev_op and dist_fn(p, prev_op) < self.threshold:
                    continue
                if (
                    doc_op_i < len(doc_ops)
                    and dist_fn(p, doc_ops[doc_op_i]) < self.threshold
                ):
                    continue
                postings.append(p)
            if len(postings) > 0:
                yield CaptionIndex.Document(id=d.id, postings=postings)


DEFAULT_AND_THRESH = 15
DEFAULT_NOT_THRESH = 15


class _QueryParser(NodeVisitor):

    def __init__(self, constants={}):
        self.grammar = GRAMMAR
        self._constants = constants

    visit_more_and = visit_more_or = visit_more_not = visit_more_tokens = \
        lambda a, b, c: c

    def visit_expr_root(self, node, children):
        assert len(children) == 3
        return children[1]

    def visit_expr_group(self, node, children):
        assert len(children) == 1
        return children[0]

    def visit_expr(self, node, children):
        assert len(children) == 1
        return children[0]

    def visit_expr_paren(self, node, children):
        assert len(children) == 7
        return children[3]

    def visit_and(self, node, children):
        assert len(children) == 3
        if children[2] is None:
            threshold = self._constants.get(
                'and_threshold', DEFAULT_AND_THRESH)
            threshold_type = 't'
        else:
            threshold_type, threshold = children[2]
            assert isinstance(threshold, int)
            assert isinstance(threshold_type, str)
        return _And([children[0], *children[1]], threshold, threshold_type)

    def visit_or(self, node, children):
        assert len(children) == 2
        return _Or([children[0], *children[1]], None, None)

    def visit_not(self, node, children):
        assert len(children) == 3
        if children[2] is None:
            threshold = self._constants.get(
                'not_threshold', DEFAULT_NOT_THRESH)
            threshold_type = 't'
        else:
            threshold_type, threshold = children[2]
            assert isinstance(threshold, int)
            assert isinstance(threshold_type, str)
        return _Not([children[0], *children[1]], threshold, threshold_type)

    def visit_threshold(self, node, children):
        assert len(children) == 4
        if children[1] == '//':
            return ('w', children[3])
        elif children[1] == '::':
            return ('t', children[3])
        else:
            raise Exception('Invalid threshold token')

    def visit_threshold_type(self, node, children):
        return node.text

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
        return [*children[0], *[t for c in children[1] for t in c]]

    def visit_token(self, node, children):
        tokenizer = default_tokenizer()
        tokens = tokenizer.tokens(node.text)
        return [_Phrase.Token(t, False) for t in tokens]

    def generic_visit(self, node, children):
        return children[-1] if children else None


class Query:
    """Parse and execute queries"""

    def __init__(self, raw_query: str, **config):
        self._tree = _QueryParser(config).parse(raw_query)

    def execute(
        self, lexicon: Lexicon, index: CaptionIndex, documents=None,
        ignore_word_not_found=True, case_insensitive=False
    ) -> Iterable[CaptionIndex.Document]:
        return self._tree.eval(_Expr.Context(
            lexicon, index, documents, ignore_word_not_found,
            case_insensitive))

    def estimate_cost(self, lexicon: Lexicon) -> float:
        return self._tree.estimate_cost(lexicon)
