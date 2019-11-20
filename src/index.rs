/* Memory mapped caption index */

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions;
use std::collections::BTreeMap;
use std::cmp;
use std::cmp::Ordering;
use std::mem;
use std::fs::File;
use memmap::{MmapOptions, Mmap};

use common::*;

// Start, End, Position, Length
type Posting = (Seconds, Seconds, Position, usize);

struct Document {
    base_offset: usize,

    // Length in milliseconds
    duration: Millis,

    // Offset of the document lexicon
    lexicon_offset: usize,
    unique_token_count: u32,

    // Offset of the inverted index
    inv_index_offset: usize,
    posting_count: u32,

    // Offset of the time inteval index
    time_index_offset: usize,
    time_int_count: u32,

    // Offset of the raw tokens
    tokens_offset: usize,
    length: usize,
}

fn parse_index(m: &Mmap, datum_size: usize, start_time_size: usize, end_time_size: usize,
               debug: bool) -> BTreeMap<DocumentId, Document> {
    let mut docs = BTreeMap::new();

    let u32_size = mem::size_of::<u32>();
    let lexicon_entry_size = 2 * datum_size;
    let posting_size = datum_size + start_time_size + end_time_size;
    let time_int_entry_size = datum_size + start_time_size + end_time_size;
    let lexicon_offset = 6 * u32_size;

    let index_size: usize = m.len();
    let mut curr_offset: usize = 0;
    while curr_offset < index_size {
        let base_offset = curr_offset;
        let doc_id: u32 = read_mmap_u32(m, base_offset);
        let duration: Millis = read_mmap_u32(m, base_offset + u32_size);
        let unique_token_count: u32 = read_mmap_u32(m, base_offset + 2 * u32_size);
        let posting_count: u32 = read_mmap_u32(m, base_offset + 3 * u32_size);
        let time_int_count: u32 = read_mmap_u32(m, base_offset + 4 * u32_size);
        let length = read_mmap_u32(m, base_offset + 5 * u32_size) as usize;

        let inv_index_offset = lexicon_offset + (unique_token_count as usize) * lexicon_entry_size;
        let time_index_offset = inv_index_offset + (posting_count as usize) * posting_size;
        let tokens_offset = time_index_offset + (time_int_count as usize) * time_int_entry_size;
        let doc_idx_len = tokens_offset + length * datum_size;

        if debug {
            eprintln!(
                "Document: id={} offset={} size={} duration={} words={} postings={} intervals={} length={}",
                doc_id, base_offset, doc_idx_len, duration, unique_token_count, posting_count,
                time_int_count, length);
        }

        docs.insert(doc_id, Document {
            base_offset: base_offset, duration: duration,
            lexicon_offset: lexicon_offset, unique_token_count: unique_token_count,
            inv_index_offset: inv_index_offset, posting_count: posting_count,
            time_index_offset: time_index_offset, time_int_count: time_int_count,
            tokens_offset: tokens_offset, length: length
        });
        curr_offset += doc_idx_len;
    }
    if debug {
        eprintln!("Loaded index containing {} documents", docs.len());
    }
    assert!(curr_offset == index_size, "Incorrect byte offsets");
    docs
}

struct _RsCaptionIndex {
    docs: BTreeMap<DocumentId, Document>,
    data: Mmap,
    datum_size: usize,
    start_time_size: usize,
    end_time_size: usize,
}

impl _RsCaptionIndex {

    fn time_int_size(&self) -> usize {
        self.start_time_size + self.end_time_size
    }

    fn posting_size(&self) -> usize {
        self.time_int_size() + self.datum_size
    }

    fn read_datum(&self, i: usize) -> u32 {
        read_mmap(&self.data, i, self.datum_size)
    }

    fn read_time_int(&self, i: usize) -> (Millis, Millis) {
        let start = read_mmap(&self.data, i, self.start_time_size);
        let diff = read_mmap(&self.data, i + self.start_time_size, self.end_time_size);
        (start, start + diff)
    }

    fn lookup_postings(&self, d: &Document, token: TokenId) -> Option<(usize, u32)> {
        let mut min_idx = 0;
        let mut max_idx = d.unique_token_count as usize;
        let token_entry_size = 2 * self.datum_size;
        let base_lexicon_offset =  d.base_offset + d.lexicon_offset;
        loop {
            if min_idx == max_idx {
                return None;
            }
            let pivot = (min_idx + max_idx) / 2;
            let ofs = pivot * token_entry_size + base_lexicon_offset;
            let pivot_token = self.read_datum(ofs);
            if pivot_token == token {
                let posting_idx = self.read_datum(ofs + self.datum_size);
                let n = if pivot < (d.unique_token_count as usize) - 1 {
                    // Get start of next entry
                    self.read_datum(ofs + token_entry_size + self.datum_size)
                } else {
                    d.posting_count
                } - posting_idx;
                assert!(n > 0, "Invalid next token posting index");
                return Some((posting_idx as usize, n))
            } else if pivot_token < token {
                min_idx = pivot + 1;
            } else {
                max_idx = pivot;
            }
        }
    }

    fn read_postings(&self, d: &Document, idx: usize, n: u32) -> Vec<Posting> {
        assert!((idx as u32) + n <= d.posting_count, "Index + n exceeds total postings");
        let time_int_size = self.time_int_size();
        let posting_size = self.posting_size();
        let base_ofs = d.base_offset + d.inv_index_offset;

        let mut postings = Vec::with_capacity(n as usize);
        for i in 0..(n as usize) {
            let ofs = (idx + i) * posting_size + base_ofs;
            let time_int = self.read_time_int(ofs);
            let pos = self.read_datum(ofs + time_int_size) as usize;
            postings.push((ms_to_s(time_int.0), ms_to_s(time_int.1), pos, 1))
        }
        postings
    }

    fn lookup_time_int(&self, d: &Document, ms: Millis) -> Option<usize> {
        let mut min_idx: usize = 0;
        let mut max_idx = d.time_int_count as usize;
        let base_index_ofs = d.base_offset + d.time_index_offset;
        let posting_size = self.posting_size();

        while max_idx > min_idx {
            let pivot: usize = (min_idx + max_idx) / 2;
            let pivot_int = self.read_time_int(base_index_ofs + pivot * posting_size);
            if ms < pivot_int.0 {
                max_idx = pivot;
            } else if ms > pivot_int.1 {
                min_idx = pivot + 1;
            } else {
                max_idx = pivot;
                min_idx = pivot;
            }
        }
        if min_idx == (d.time_int_count as usize) {
            None
        } else {
            Some(min_idx)
        }
    }

    fn lookup_time_int_by_idx(
        &self, d: &Document, idx: u32, hint_idx: usize
    ) -> Option<(usize, (Millis, Millis))> {
        let mut min_idx: usize = hint_idx;
        let mut max_idx = d.time_int_count as usize;
        let base_index_ofs = d.base_offset + d.time_index_offset;
        let posting_size = self.posting_size();
        let time_int_size = self.time_int_size();

        while max_idx > min_idx {
            let pivot: usize = (min_idx + max_idx) / 2;
            let pivot_idx = self.read_datum(
                base_index_ofs + pivot * posting_size + time_int_size);
            if pivot + 1 == d.time_int_count as usize {
                return if idx >= pivot_idx {
                    Some((pivot, self.read_time_int(base_index_ofs + pivot * posting_size)))
                } else { None };
            }
            let pivot_idx_next = self.read_datum(
                base_index_ofs + (pivot + 1) * posting_size + time_int_size);
            assert!(pivot_idx_next >= pivot_idx, "Non-increasing idx: {} < {}",
                    pivot_idx_next, pivot_idx);
            if idx < pivot_idx {
                max_idx = pivot;
            } else if idx >= pivot_idx && idx < pivot_idx_next {
                return Some((pivot, self.read_time_int(base_index_ofs + pivot * posting_size)));
            } else {
                min_idx = pivot + 1;
            }
        }
        None
    }

    fn check_is_ngram(&self, ngram: &Vec<TokenIdOneOf>, anchor_idx: usize, document: &Document,
                      ngram_base_pos: usize) -> bool {
        for k in 0..ngram.len() {
            if k != anchor_idx {
                let token_k = self.read_datum(
                    (ngram_base_pos + k) * self.datum_size
                    + document.base_offset + document.tokens_offset);

                // Move on to the next anchor token location
                if !ngram[k].contains(&token_k) {
                    return false;
                }
            }
        }
        true
    }

    fn lookup_ngram_contains(
        &self, ngram: &Vec<TokenIdOneOf>, anchor_idx: usize, document: &Document
    ) -> bool {
        let base_index_ofs: usize = document.base_offset + document.inv_index_offset;
        let time_int_size = self.time_int_size();
        let posting_size = self.posting_size();
        let ngram_len = ngram.len();

        let anchor_idx_posting_offsets: Vec<(usize, u32)> =
            ngram[anchor_idx].iter().filter_map(
                |token| self.lookup_postings(document, *token)
            ).collect();
        if anchor_idx_posting_offsets.len() == 0 {
            return false;    // None of the optons matched
        }

        for i in 0..anchor_idx_posting_offsets.len() {
            let anchor_token_posting_idx = anchor_idx_posting_offsets[i].0;
            let anchor_token_posting_count = anchor_idx_posting_offsets[i].1 as usize;

            // Look for ngram around anchor index
            for j in 0..anchor_token_posting_count {
                let ngram_anchor_pos = self.read_datum(
                    base_index_ofs +
                    (anchor_token_posting_idx + j) * posting_size +
                    time_int_size
                ) as usize;

                if ngram_anchor_pos < anchor_idx ||
                   ngram_anchor_pos - anchor_idx + ngram_len > document.length {
                    continue;
                }

                // Check indices around anchor index
                let ngram_base_pos = ngram_anchor_pos - anchor_idx;
                if self.check_is_ngram(ngram, anchor_idx, document, ngram_base_pos) {
                    return true;
                }
            }
        }
        false
    }

    fn lookup_and_load_ngrams(
        &self, ngram: &Vec<TokenIdOneOf>, anchor_idx: usize, document: &Document
    ) -> Option<Vec<Posting>> {
        let base_index_ofs: usize = document.base_offset + document.inv_index_offset;
        let time_int_size = self.time_int_size();
        let posting_size = self.posting_size();
        let ngram_len = ngram.len();

        let anchor_idx_posting_offsets: Vec<(usize, u32)> = ngram[anchor_idx].iter().filter_map(
            |token| self.lookup_postings(document, *token)
        ).collect();
        if anchor_idx_posting_offsets.len() == 0 {
            return None;    // None of the optons matched
        }

        let mut postings: Vec<Posting> = vec![];
        let mut anchors_used = 0;
        for i in 0..anchor_idx_posting_offsets.len() {
            let mut anchor_used = false;
            let anchor_token_posting_idx = anchor_idx_posting_offsets[i].0;
            let anchor_token_posting_count = anchor_idx_posting_offsets[i].1 as usize;
            let mut hint_time_int_idx: usize = 0;

            // Look for ngram around anchor index
            for j in 0..anchor_token_posting_count {
                let ngram_anchor_pos = self.read_datum(
                    base_index_ofs +
                    (anchor_token_posting_idx + j) * posting_size +
                    time_int_size
                ) as usize;

                if ngram_anchor_pos < anchor_idx ||
                   ngram_anchor_pos - anchor_idx + ngram_len > document.length {
                    continue;
                }

                // Check indices around anchor index
                let ngram_base_pos = ngram_anchor_pos - anchor_idx;
                if self.check_is_ngram(&ngram, anchor_idx, document, ngram_base_pos) {
                    // All other indices matched
                    let ngram_anchor_time_int = self.read_time_int(
                        base_index_ofs +
                        (anchor_token_posting_idx + j) * posting_size);
                    let start_ms = if anchor_idx != 0 {
                        match self.lookup_time_int_by_idx(
                            document, ngram_base_pos as u32, hint_time_int_idx
                        ) {
                            Some((new_hint_idx, ngram_start_time_int)) => {
                                hint_time_int_idx = new_hint_idx;
                                cmp::min(ngram_start_time_int.0, ngram_anchor_time_int.0)
                            },
                            None => ngram_anchor_time_int.0
                        }
                    } else {
                        ngram_anchor_time_int.0
                    };
                    let end_ms = if anchor_idx + 1 == ngram_len {
                        ngram_anchor_time_int.1
                    } else {
                        match self.lookup_time_int_by_idx(
                            document, (ngram_base_pos + ngram_len - 1) as u32, hint_time_int_idx
                        ) {
                            Some((_, ngram_end_time_int)) => {
                                cmp::max(ngram_end_time_int.1, ngram_anchor_time_int.1)
                            },
                            None => ngram_anchor_time_int.1
                        }
                    };

                    postings.push((
                        ms_to_s(start_ms), ms_to_s(end_ms), ngram_base_pos, ngram_len
                    ));
                    anchor_used |= true;
                }
            }

            // Need to resort if multiple anchor words used
            if anchor_used {
                anchors_used += 1;
            }
        }
        if postings.len() > 0 {
            if anchors_used > 1 {
                postings.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
                    Some(ord) => if ord == Ordering::Equal {a.2.cmp(&b.2)} else {ord}
                    None => a.2.cmp(&b.2)
                });
            }
            Some(postings)
        } else {
            None
        }
    }
}

#[pyclass]
pub struct RsCaptionIndex {
    _internal: _RsCaptionIndex,
    debug: bool
}

#[pymethods]
impl RsCaptionIndex {

    fn document_exists(&self, doc_id: DocumentId) -> PyResult<bool> {
        Ok(self._internal.docs.get(&doc_id).is_some())
    }

    fn document_length(&self, doc_id: DocumentId) -> PyResult<(usize, f32)> {
        match self._internal.docs.get(&doc_id) {
            Some(d) => Ok((d.length, ms_to_s(d.duration))),
            None => Err(exceptions::ValueError::py_err("Document not found"))
        }
    }

    fn unigram_search(
        &self, unigram: TokenIdOneOf, mut doc_ids: Vec<DocumentId>
    ) -> PyResult<Vec<(DocumentId, Vec<Posting>)>> {
        if self.debug {
            let len_str = doc_ids.len().to_string();
            eprintln!("unigram search: [{:?}] in {} documents", unigram,
                      if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
        }
        let lookup_and_read_postings = |d| {
            let mut unique_unigrams = 0;
            let mut postings: Vec<Posting> = unigram.iter().flat_map(
                |token| match self._internal.lookup_postings(d, *token) {
                    Some(p) => {
                        unique_unigrams += 1;
                        self._internal.read_postings(d, p.0, p.1)
                    },
                    None => Vec::new()
                }
            ).collect();
            if postings.len() > 0 {
                if unique_unigrams > 1 {
                    postings.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
                        Some(ord) => if ord == Ordering::Equal {a.2.cmp(&b.2)} else {ord}
                        None => a.2.cmp(&b.2)
                    });
                }
                Some(postings)
            } else {
                None
            }
        };
        let docs_to_unigrams =
            if doc_ids.len() > 0 {
                doc_ids.par_sort();
                doc_ids.par_iter().filter_map(
                    |id| match self._internal.docs.get(&id) {
                        None => None,
                        Some(d) => match lookup_and_read_postings(d) {
                            None => None,
                            Some(p) => Some((*id, p))
                        }
                    }
                ).collect()
            } else {
                self._internal.docs.par_iter().filter_map(
                    |(id, d)| match lookup_and_read_postings(d) {
                        None => None,
                        Some(p) => Some((*id, p))
                    }
                ).collect()
            };
        Ok(docs_to_unigrams)
    }

    fn unigram_contains(
        &self, unigram: TokenIdOneOf, doc_ids: Vec<DocumentId>
    ) -> PyResult<Vec<DocumentId>> {
        if self.debug {
            let len_str = doc_ids.len().to_string();
            eprintln!("unigram contains: [{:?}] in {} documents", unigram,
                      if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
        }
        let has_unigram = |d| {
            for i in 0..unigram.len() {
                if self._internal.lookup_postings(d, unigram[i]).is_some() {
                    return true;
                }
            }
            false
        };
        let docs_w_token =
            if doc_ids.len() > 0 {
                doc_ids.par_iter().filter_map(
                    |id| match self._internal.docs.get(&id) {
                        None => None,
                        Some(d) => if has_unigram(d) {Some(*id)} else {None}
                    }
                ).collect()
            } else {
                self._internal.docs.par_iter().filter_map(
                    |(id, d)| if has_unigram(d) {Some(*id)} else {None}
                ).collect()
            };
        Ok(docs_w_token)
    }

    fn ngram_search(
        &self, ngram: Vec<TokenIdOneOf>, mut doc_ids: Vec<DocumentId>, anchor_idx: usize
    ) -> PyResult<Vec<(DocumentId, Vec<Posting>)>> {
        if ngram.len() <= 1 {
            Err(exceptions::ValueError::py_err("Ngram must have at least 2 tokens"))
        } else {
            if self.debug {
                let len_str = doc_ids.len().to_string();
                eprintln!("ngram search: {:?} in {} documents", ngram,
                          if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
            }
            let load_ngrams = |d: &Document| -> Option<Vec<Posting>> {
                self._internal.lookup_and_load_ngrams(&ngram, anchor_idx, d)
            };
            let docs_to_ngrams =
                if doc_ids.len() > 0 {
                    doc_ids.par_sort();
                    doc_ids.par_iter().filter_map(
                        |id| match self._internal.docs.get(&id) {
                            None => None,
                            Some(d) => match load_ngrams(d) {
                                None => None,
                                Some(p) => Some((*id, p))
                            }
                        }
                    ).collect()
                } else {
                    self._internal.docs.par_iter().filter_map(
                        |(id, d)| match load_ngrams(d) {
                            None => None,
                            Some(p) => Some((*id, p))
                        }
                    ).collect()
                };
            Ok(docs_to_ngrams)
        }
    }

    fn ngram_contains(
        &self, ngram: Vec<TokenIdOneOf>, doc_ids: Vec<DocumentId>, anchor_idx: usize
    ) -> PyResult<Vec<DocumentId>> {
        if ngram.len() <= 1 {
            Err(exceptions::ValueError::py_err("Ngram must have at least 2 tokens"))
        } else {
            if self.debug {
                let len_str = doc_ids.len().to_string();
                eprintln!("ngram contains: {:?} in {} documents", ngram,
                          if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
            }
            let has_ngram = |d: &Document| -> bool {
                self._internal.lookup_ngram_contains(&ngram, anchor_idx, d)
            };
            let docs_w_ngram =
                if doc_ids.len() > 0 {
                    doc_ids.par_iter().filter_map(
                        |id| match self._internal.docs.get(&id) {
                             None => None,
                             Some(d) => if has_ngram(d) {Some(*id)} else {None}
                        }
                    ).collect()
                } else {
                    self._internal.docs.par_iter().filter_map(
                        |(id, d)| if has_ngram(d) {Some(*id)} else {None}
                    ).collect()
                };
            Ok(docs_w_ngram)
        }
    }

    fn tokens(&self, doc_id: DocumentId, position: usize, n: usize) -> PyResult<Vec<TokenId>> {
        if self.debug {
            eprintln!("tokens: {}+{} in {}", position, n, doc_id);
        }
        match self._internal.docs.get(&doc_id) {
            Some(d) => {
                let min_pos = cmp::min(position, d.length);
                let max_pos = cmp::min(position + n, d.length);
                let mut tokens = Vec::with_capacity(max_pos - min_pos);
                for pos in min_pos..max_pos {
                    let ofs = pos * self._internal.datum_size + d.base_offset + d.tokens_offset;
                    tokens.push(self._internal.read_datum(ofs));
                }
                Ok(tokens)
            },
            None => Err(exceptions::ValueError::py_err("Document not found"))
        }
    }

    fn intervals(&self, doc_id: DocumentId, start: Seconds, end: Seconds) ->
                 PyResult<Vec<Posting>> {
        if self.debug {
            eprintln!("intervals: {}s to {}s in {}", start, end, doc_id);
        }
        // Get document locations that overlap start and end
        if start > ms_to_s(u32::max_value()) {
            return Err(exceptions::ValueError::py_err("Start time exceeds maximum allowed"))
        }
        let start_ms = if start > 0. {s_to_ms(start)} else {0};
        let posting_size = self._internal.posting_size();
        let time_int_size = self._internal.time_int_size();
        match self._internal.docs.get(&doc_id) {
            Some(d) => {
                let mut locations = vec![];
                match self._internal.lookup_time_int(d, start_ms) {
                    Some(start_idx_immut) => {
                        let mut start_idx = start_idx_immut;
                        if start_idx > 0 {
                            start_idx -= 1;
                        }
                        let end_ms = if ms_to_s(d.duration) < end {d.duration} else {s_to_ms(end)};
                        let base_index_ofs = d.base_offset + d.time_index_offset;
                        for i in start_idx..(d.time_int_count as usize) {
                            let ofs = i * posting_size + base_index_ofs;
                            let time_int = self._internal.read_time_int(ofs);
                            if cmp::min(end_ms, time_int.1) >= cmp::max(start_ms, time_int.0) {
                                // Non-zero overlap
                                let pos = self._internal.read_datum(ofs + time_int_size) as usize;
                                let next_pos: usize = if i + 1 < (d.time_int_count as usize) {
                                    self._internal.read_datum(
                                        ofs + posting_size + time_int_size) as usize
                                } else {d.length};
                                assert!(next_pos >= pos, "postions are not non-decreasing");
                                locations.push(
                                    (ms_to_s(time_int.0), ms_to_s(time_int.1), pos, next_pos - pos))
                            }
                            if time_int.0 > end_ms {
                                break;
                            }
                        }
                    },
                    None => ()
                };
                Ok(locations)
            },
            None => Err(exceptions::Exception::py_err("Document not found"))
        }
    }

    fn position(&self, doc_id: DocumentId, time: Seconds) -> PyResult<Position> {
        if self.debug {
            eprintln!("position: {}s in {}", time, doc_id);
        }
        match self._internal.docs.get(&doc_id) {
            Some(d) => Ok({
                match self._internal.lookup_time_int(d, s_to_ms(time)) {
                    Some(idx) => {
                        let ofs = d.base_offset + d.time_index_offset +
                            idx * self._internal.posting_size() + self._internal.time_int_size();
                        self._internal.read_datum(ofs) as Position
                    },
                    None => d.length as Position
                }
            }),
            None => Err(exceptions::ValueError::py_err("Document not found"))
        }
    }

    #[new]
    unsafe fn __new__(obj: &PyRawObject, index_file: String, datum_size: usize,
                      start_time_size: usize, end_time_size: usize, debug: bool
    ) -> PyResult<()> {
        let mmap = MmapOptions::new().map(&File::open(&index_file)?);
        match mmap {
            Ok(m) => {
                let docs = parse_index(&m, datum_size, start_time_size, end_time_size, debug);
                obj.init(RsCaptionIndex {
                    _internal: _RsCaptionIndex {
                        docs: docs, data: m, datum_size: datum_size,
                        start_time_size: start_time_size, end_time_size: end_time_size
                    },
                    debug: debug
                });
                Ok(())
            },
            Err(s) => Err(exceptions::Exception::py_err(s.to_string()))
        }
    }
}
