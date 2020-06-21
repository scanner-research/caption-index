/* Memory mapped caption index */

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions;
use std::collections::BTreeMap;
use std::cmp;
use std::cmp::Ordering;
use std::mem;
use std::fs::{File, metadata, read_dir};
use memmap::{MmapOptions, Mmap};

use common::*;

struct Document {
    // The file containing the document index
    file_num: usize,

    // Byte offset in file
    base_offset: usize,

    // Offset of the document lexicon
    lexicon_offset: usize,
    unique_token_count: u32,

    // Offset of the inverted index
    inv_index_offset: usize,
    posting_count: u32,
}

fn parse_index(m: &Mmap, file_num: usize, datum_size: usize,
               start_time_size: usize, end_time_size: usize, debug: bool
) -> BTreeMap<DocumentId, Document> {
    let mut docs = BTreeMap::new();

    let u32_size = mem::size_of::<u32>();
    let lexicon_entry_size = 2 * datum_size;
    let posting_size = datum_size + start_time_size + end_time_size;
    let lexicon_offset = 3 * u32_size;

    let index_size: usize = m.len();
    let mut curr_offset: usize = 0;
    while curr_offset < index_size {
        let base_offset = curr_offset;
        let doc_id: u32 = read_mmap_u32(m, base_offset);
        let unique_token_count: u32 = read_mmap_u32(m, base_offset + u32_size);
        let posting_count: u32 = read_mmap_u32(m, base_offset + 2 * u32_size);

        let inv_index_offset = lexicon_offset + (unique_token_count as usize) * lexicon_entry_size;
        let doc_index_len = inv_index_offset + (posting_count as usize) * posting_size;

        if debug {
            eprintln!("Document: id={} offset={} words={} postings={}",
                      doc_id, base_offset, unique_token_count, posting_count);
        }

        docs.insert(doc_id, Document {
            file_num: file_num, base_offset: base_offset,
            lexicon_offset: lexicon_offset, unique_token_count: unique_token_count,
            inv_index_offset: inv_index_offset, posting_count: posting_count
        });
        curr_offset += doc_index_len;
    }
    if debug {
        eprintln!("Loaded index containing {} documents", docs.len());
    }
    assert!(curr_offset == index_size, "Incorrect byte offsets");
    docs
}

struct _RsCaptionIndex {
    docs: BTreeMap<DocumentId, Document>,
    data: Vec<Mmap>,
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

    fn read_datum(&self, m: &Mmap, i: usize) -> u32 {
        read_mmap(m, i, self.datum_size)
    }

    fn read_time_int(&self, m: &Mmap, i: usize) -> (Millis, Millis) {
        let start = read_mmap(m, i, self.start_time_size);
        let diff = read_mmap(m, i + self.start_time_size, self.end_time_size);
        (start, start + diff)
    }

    fn lookup_postings(&self, d: &Document, token: TokenId) -> Option<(usize, u32)> {
        let m = &self.data[d.file_num];
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
            let pivot_token = self.read_datum(m, ofs);
            if pivot_token == token {
                let posting_idx = self.read_datum(m, ofs + self.datum_size);
                let n = if pivot < (d.unique_token_count as usize) - 1 {
                    // Get start of next entry
                    self.read_datum(m, ofs + token_entry_size + self.datum_size)
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
        let m = &self.data[d.file_num];
        let time_int_size = self.time_int_size();
        let posting_size = self.posting_size();
        let base_ofs = d.base_offset + d.inv_index_offset;

        let mut postings = Vec::with_capacity(n as usize);
        for i in 0..(n as usize) {
            let ofs = (idx + i) * posting_size + base_ofs;
            let time_int = self.read_time_int(m, ofs);
            let pos = self.read_datum(m, ofs + time_int_size) as usize;
            postings.push((ms_to_s(time_int.0), ms_to_s(time_int.1), pos, 1))
        }
        postings
    }

    // fn lookup_ngram_contains(
    //     &self, ngram: &Vec<Token>, anchor_idx: usize, document: &Document
    // ) -> bool {
    //     let m = &self.data[document.file_num];
    //     let base_index_ofs: usize = document.base_offset + document.inv_index_offset;
    //     let time_int_size = self.time_int_size();
    //     let posting_size = self.posting_size();
    //     let ngram_len = ngram.len();
    //
    //     let anchor_idx_posting_offsets: Vec<(usize, u32)> =
    //         ngram[anchor_idx].iter().filter_map(
    //             |token| self.lookup_postings(document, *token)
    //         ).collect();
    //     if anchor_idx_posting_offsets.len() == 0 {
    //         return false;    // None of the optons matched
    //     }
    //
    //     for i in 0..anchor_idx_posting_offsets.len() {
    //         let anchor_token_posting_idx = anchor_idx_posting_offsets[i].0;
    //         let anchor_token_posting_count = anchor_idx_posting_offsets[i].1 as usize;
    //
    //         // Look for ngram around anchor index
    //         for j in 0..anchor_token_posting_count {
    //             let ngram_anchor_pos = self.read_datum(
    //                 m,
    //                 base_index_ofs + (anchor_token_posting_idx + j) * posting_size + time_int_size
    //             ) as usize;
    //
    //             if ngram_anchor_pos < anchor_idx ||
    //                ngram_anchor_pos - anchor_idx + ngram_len > document.length {
    //                 continue;
    //             }
    //
    //             // Check indices around anchor index
    //             let ngram_base_pos = ngram_anchor_pos - anchor_idx;
    //             if self.check_is_ngram(ngram, anchor_idx, document, ngram_base_pos) {
    //                 return true;
    //             }
    //         }
    //     }
    //     false
    // }

    // fn lookup_and_load_ngrams(
    //     &self, ngram: &Vec<Token>, anchor_idx: usize, document: &Document
    // ) -> Option<Vec<Posting>> {
    //     let m = &self.data[document.file_num];
    //     let base_index_ofs: usize = document.base_offset + document.inv_index_offset;
    //     let time_int_size = self.time_int_size();
    //     let posting_size = self.posting_size();
    //     let ngram_len = ngram.len();
    //
    //     let anchor_idx_posting_offsets: Vec<(usize, u32)> = ngram[anchor_idx].iter().filter_map(
    //         |token| self.lookup_postings(document, *token)
    //     ).collect();
    //     if anchor_idx_posting_offsets.len() == 0 {
    //         return None;    // None of the optons matched
    //     }
    //
    //     let mut postings: Vec<Posting> = vec![];
    //     let mut anchors_used = 0;
    //     for i in 0..anchor_idx_posting_offsets.len() {
    //         let mut anchor_used = false;
    //         let anchor_token_posting_idx = anchor_idx_posting_offsets[i].0;
    //         let anchor_token_posting_count = anchor_idx_posting_offsets[i].1 as usize;
    //         let mut hint_time_int_idx: usize = 0;
    //
    //         // Look for ngram around anchor index
    //         for j in 0..anchor_token_posting_count {
    //             let ngram_anchor_pos = self.read_datum(
    //                 m,
    //                 base_index_ofs + (anchor_token_posting_idx + j) * posting_size + time_int_size
    //             ) as usize;
    //
    //             if ngram_anchor_pos < anchor_idx ||
    //                ngram_anchor_pos - anchor_idx + ngram_len > document.length {
    //                 continue;
    //             }
    //
    //             // Check indices around anchor index
    //             let ngram_base_pos = ngram_anchor_pos - anchor_idx;
    //             if self.check_is_ngram(&ngram, anchor_idx, document, ngram_base_pos) {
    //                 // All other indices matched
    //                 let ngram_anchor_time_int = self.read_time_int(
    //                     m, base_index_ofs + (anchor_token_posting_idx + j) * posting_size);
    //                 let start_ms = if anchor_idx != 0 {
    //                     match self.lookup_time_int_by_idx(
    //                         document, ngram_base_pos as u32, hint_time_int_idx
    //                     ) {
    //                         Some((new_hint_idx, ngram_start_time_int)) => {
    //                             hint_time_int_idx = new_hint_idx;
    //                             cmp::min(ngram_start_time_int.0, ngram_anchor_time_int.0)
    //                         },
    //                         None => ngram_anchor_time_int.0
    //                     }
    //                 } else {
    //                     ngram_anchor_time_int.0
    //                 };
    //                 let end_ms = if anchor_idx + 1 == ngram_len {
    //                     ngram_anchor_time_int.1
    //                 } else {
    //                     match self.lookup_time_int_by_idx(
    //                         document, (ngram_base_pos + ngram_len - 1) as u32, hint_time_int_idx
    //                     ) {
    //                         Some((_, ngram_end_time_int)) => {
    //                             cmp::max(ngram_end_time_int.1, ngram_anchor_time_int.1)
    //                         },
    //                         None => ngram_anchor_time_int.1
    //                     }
    //                 };
    //
    //                 postings.push((
    //                     ms_to_s(start_ms), ms_to_s(end_ms), ngram_base_pos, ngram_len
    //                 ));
    //                 anchor_used |= true;
    //             }
    //         }
    //
    //         // Need to resort if multiple anchor words used
    //         if anchor_used {
    //             anchors_used += 1;
    //         }
    //     }
    //     if postings.len() > 0 {
    //         if anchors_used > 1 {
    //             postings.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
    //                 Some(ord) => if ord == Ordering::Equal {a.2.cmp(&b.2)} else {ord}
    //                 None => a.2.cmp(&b.2)
    //             });
    //         }
    //         Some(postings)
    //     } else {
    //         None
    //     }
    // }
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

    fn unigram_search(
        &self, unigram: Token, mut doc_ids: Vec<DocumentId>
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
        &self, unigram: Token, doc_ids: Vec<DocumentId>
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

    // fn ngram_search(
    //     &self, ngram: Vec<Token>, mut doc_ids: Vec<DocumentId>, anchor_idx: usize
    // ) -> PyResult<Vec<(DocumentId, Vec<Posting>)>> {
    //     if ngram.len() <= 1 {
    //         Err(exceptions::ValueError::py_err("Ngram must have at least 2 tokens"))
    //     } else {
    //         if self.debug {
    //             let len_str = doc_ids.len().to_string();
    //             eprintln!("ngram search: {:?} in {} documents", ngram,
    //                       if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
    //         }
    //         let load_ngrams = |d: &Document| -> Option<Vec<Posting>> {
    //             self._internal.lookup_and_load_ngrams(&ngram, anchor_idx, d)
    //         };
    //         let docs_to_ngrams =
    //             if doc_ids.len() > 0 {
    //                 doc_ids.par_sort();
    //                 doc_ids.par_iter().filter_map(
    //                     |id| match self._internal.docs.get(&id) {
    //                         None => None,
    //                         Some(d) => match load_ngrams(d) {
    //                             None => None,
    //                             Some(p) => Some((*id, p))
    //                         }
    //                     }
    //                 ).collect()
    //             } else {
    //                 self._internal.docs.par_iter().filter_map(
    //                     |(id, d)| match load_ngrams(d) {
    //                         None => None,
    //                         Some(p) => Some((*id, p))
    //                     }
    //                 ).collect()
    //             };
    //         Ok(docs_to_ngrams)
    //     }
    // }
    //
    // fn ngram_contains(
    //     &self, ngram: Vec<Token>, doc_ids: Vec<DocumentId>, anchor_idx: usize
    // ) -> PyResult<Vec<DocumentId>> {
    //     if ngram.len() <= 1 {
    //         Err(exceptions::ValueError::py_err("Ngram must have at least 2 tokens"))
    //     } else {
    //         if self.debug {
    //             let len_str = doc_ids.len().to_string();
    //             eprintln!("ngram contains: {:?} in {} documents", ngram,
    //                       if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
    //         }
    //         let has_ngram = |d: &Document| -> bool {
    //             self._internal.lookup_ngram_contains(&ngram, anchor_idx, d)
    //         };
    //         let docs_w_ngram =
    //             if doc_ids.len() > 0 {
    //                 doc_ids.par_iter().filter_map(
    //                     |id| match self._internal.docs.get(&id) {
    //                          None => None,
    //                          Some(d) => if has_ngram(d) {Some(*id)} else {None}
    //                     }
    //                 ).collect()
    //             } else {
    //                 self._internal.docs.par_iter().filter_map(
    //                     |(id, d)| if has_ngram(d) {Some(*id)} else {None}
    //                 ).collect()
    //             };
    //         Ok(docs_w_ngram)
    //     }
    // }

    #[new]
    unsafe fn __new__(obj: &PyRawObject, index_path: String, datum_size: usize,
                      start_time_size: usize, end_time_size: usize, debug: bool
    ) -> PyResult<()> {
        let mut index_files = vec![];
        match metadata(index_path.clone()) {
            Ok(meta) => {
                if meta.is_dir() {
                    let index_paths = read_dir(index_path.clone()).unwrap();
                    for entry in index_paths {
                        let fname = entry.unwrap().path().file_name().unwrap().to_string_lossy().into_owned();
                        let mut fpath = index_path.clone();
                        fpath.push_str("/");
                        fpath.push_str(&fname);
                        index_files.push(fpath.clone());
                    }
                } else {
                    index_files.push(index_path.clone());
                }
            },
            Err(_e) => return Err(exceptions::OSError::py_err("Unable to stat index files"))
        }

        let index_mmaps: Vec<Mmap> = index_files.iter().map(|index_path| {
            MmapOptions::new().map(&File::open(&index_path).unwrap()).unwrap()
        }).collect();

        let mut docs = BTreeMap::new();
        for i in 0..index_mmaps.len() {
            let chunk_docs = parse_index(
                &index_mmaps[i], i, datum_size, start_time_size, end_time_size, debug);
            docs.extend(chunk_docs);
        }

        obj.init(RsCaptionIndex {
            _internal: _RsCaptionIndex {
                docs: docs, data: index_mmaps, datum_size: datum_size,
                start_time_size: start_time_size, end_time_size: end_time_size
            },
            debug: debug
        });
        Ok(())
    }
}
