import re
import pysrt
import pyvtt
import traceback
from html.parser import HTMLParser
from collections import defaultdict, deque, Counter
from io import BytesIO
from typing import Callable, Dict, List, Tuple, NamedTuple

from .index import Lexicon, BinaryFormat
from .tokenize import Tokenizer, default_tokenizer


def __millis_to_seconds(t: int) -> float:
    return t / 1000


class CaptionLine(NamedTuple):
    start: int
    end: int
    text: str


class CaptionParseError(Exception):
    pass


def __load_srt(doc_path: str) -> List[CaptionLine]:
    try:
        subs = pysrt.open(doc_path)
    except:
        try:
            subs = pysrt.open(doc_path, encoding='iso-8859-1')
        except:
            raise CaptionParseError('Cannot parse {}'.format(doc_path))
    return [CaptionLine(s.start.ordinal, s.end.ordinal, s.text) for s in subs]


def __strip_ml(s: str) -> str:
    return HTMLParser().unescape(re.sub('<[^<]+?>', '', s))


def __load_vtt(doc_path: str) -> List[CaptionLine]:
    try:
        subs = pyvtt.open(doc_path)
    except:
        raise CaptionParseError('Cannot parse {}'.format(doc_path))
    return [CaptionLine(s.start.ordinal, s.end.ordinal, __strip_ml(s.text))
            for s in subs]


def load_file(doc_path: str) -> List[CaptionLine]:
    if doc_path.endswith('.vtt'):
        return __load_vtt(doc_path)
    else:
        return __load_srt(doc_path)


def get_document_word_counts(
    doc_path: str,
    tokenizer: Tokenizer = default_tokenizer(),
    max_word_len: int = 1000
) -> Counter:
    words = Counter()
    try:
        subs = load_file(doc_path)
    except Exception as e:
        print(e)
        return words
    for s in subs:
        tokens = tokenizer.tokens(s.text)
        words.update(t for t in tokens if len(t) <= max_word_len)
    return words


def __read_and_index_document(
    doc_path: str, lexicon: Lexicon, tokenizer: Tokenizer,
    binary_format: BinaryFormat
):
    doc_inv_index = defaultdict(deque)  # token_id -> [postings]
    doc_lines = deque()                 # [(position, start, end, [tokens])]
    try:
        subs = load_file(doc_path)
        doc_position = 0
        for s in subs:
            start, end = s.start, s.end
            if start > end:
                print('Warning: start time > end time ({} > {})'.format(
                      start, end))
                end = start
            if end - start > binary_format.max_time_interval:
                print('Warning: end - start > {}ms'.format(
                      binary_format.max_time_interval))
                end = start + binary_format.max_time_interval

            tokens = deque()
            entry_start_position = doc_position

            for t in tokenizer.tokens(s.text):
                token = None
                try:
                    try:
                        token = lexicon[t]
                    except Lexicon.WordDoesNotExist:
                        print('Unknown token: {}'.format(t))
                        continue
                    doc_inv_index[token.id].append((doc_position, start, end))
                finally:
                    doc_position += 1
                    tokens.append(
                        binary_format.max_datum_value
                        if token is None else token.id)

            doc_lines.append((entry_start_position, start, end, tokens))

        if len(doc_lines) == 0:
            print('Empty file: {}'.format(doc_path))
    except Exception as e:
        print('Failed to index: {}'.format(doc_path))
        traceback.print_exc()
    return doc_inv_index, doc_lines


InvertedIndexEntry = Tuple[int, int, int]
DocumentLine = Tuple[int, int, int, List[int]]


def __write_index_for_document(
    doc_id: int, doc_inv_index: Dict[int, InvertedIndexEntry],
    doc_lines: List[DocumentLine], binary_format: BinaryFormat,
    out_path: str
):
    f_tokens = BytesIO()
    f_inv_index = BytesIO()

    doc_unique_token_count = len(doc_inv_index)
    doc_line_count = len(doc_lines)

    doc_posting_count = 0
    for token_id in sorted(doc_inv_index):
        f_tokens.write(binary_format.encode_datum(token_id))
        f_tokens.write(binary_format.encode_datum(doc_posting_count))

        postings = doc_inv_index[token_id]
        assert len(postings) > 0

        for (position, start, end) in postings:
            f_inv_index.write(binary_format.encode_time_interval(start, end))
            f_inv_index.write(binary_format.encode_datum(position))
            doc_posting_count += 1

    f_time_index = BytesIO()
    for position, start, end, _ in doc_lines:
        f_time_index.write(binary_format.encode_time_interval(start, end))
        f_time_index.write(binary_format.encode_datum(position))

    doc_len = 0
    doc_duration = 0
    f_data = BytesIO()
    for _, _, end, tokens in doc_lines:
        for t in tokens:
            f_data.write(binary_format.encode_datum(t))
        doc_len += len(tokens)
        doc_duration = max(doc_duration, end)

    # Checks to make sure that the lengths are correct
    assert doc_unique_token_count == f_tokens.tell() / (
        2 * binary_format.datum_bytes)
    assert doc_posting_count == f_inv_index.tell() / (
        binary_format.datum_bytes + binary_format.time_interval_bytes)
    assert doc_line_count == f_time_index.tell() / (
        binary_format.datum_bytes + binary_format.time_interval_bytes)
    assert doc_len == f_data.tell() / binary_format.datum_bytes

    # Write the index for the single document
    with open(out_path, 'wb') as f:
        f.write(binary_format.encode_u32(doc_id))
        f.write(binary_format.encode_u32(doc_duration))
        f.write(binary_format.encode_u32(doc_unique_token_count))
        f.write(binary_format.encode_u32(doc_posting_count))
        f.write(binary_format.encode_u32(doc_line_count))
        f.write(binary_format.encode_u32(doc_len))
        f.write(f_tokens.getvalue())
        f.write(f_inv_index.getvalue())
        f.write(f_time_index.getvalue())
        f.write(f_data.getvalue())


def index_document(
    doc_id: int, doc_path: str, lexicon: Lexicon, out_path: str,
    tokenizer: Tokenizer = default_tokenizer(),
    binary_format: BinaryFormat = BinaryFormat.default()
):
    doc_inv_index, doc_lines = __read_and_index_document(
        doc_path, lexicon, tokenizer, binary_format)
    __write_index_for_document(
        doc_id, doc_inv_index, doc_lines, binary_format, out_path)
