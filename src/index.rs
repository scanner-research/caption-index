/* Memory mapped caption index */

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions;
use std::collections::{BTreeMap, BinaryHeap};
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

#[derive(Copy, Clone)]
struct HeapPosting {
    posting: Posting,
    origin: usize
}

impl Eq for HeapPosting {}

impl PartialEq for HeapPosting {
    fn eq(&self, other: &Self) -> bool {
        self.posting.2 == other.posting.2 && self.origin == other.origin
    }
}

impl Ord for HeapPosting {
    fn cmp(&self, other: &HeapPosting) -> Ordering {
        other.posting.2.cmp(&self.posting.2).then_with(|| self.origin.cmp(&other.origin))
    }
}

impl PartialOrd for HeapPosting {
    fn partial_cmp(&self, other: &HeapPosting) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct _RsCaptionIndexImpl {
    docs: BTreeMap<DocumentId, Document>,
    data: Vec<Mmap>,
    datum_size: usize,
    start_time_size: usize,
    end_time_size: usize,
}

impl _RsCaptionIndexImpl {

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

    fn lookup_posting_offsets_one(&self, d: &Document, token: TokenId) -> Option<(usize, u32)> {
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
                let posting_idx_plus_n = if pivot < (d.unique_token_count as usize) - 1 {
                    // Get start of next entry
                    self.read_datum(m, ofs + token_entry_size + self.datum_size)
                } else {
                    d.posting_count
                };
                assert!(posting_idx_plus_n > posting_idx, "Invalid next token posting index");
                return Some((posting_idx as usize, posting_idx_plus_n - posting_idx))
            } else if pivot_token < token {
                min_idx = pivot + 1;
            } else {
                max_idx = pivot;
            }
        }
    }

    fn lookup_posting_offsets_many(&self, document: &Document, token: &Token) -> Option<Vec<(usize, u32)>> {
        let posting_offsets: Vec<(usize, u32)> =
            token.iter().filter_map(
                |token_id| self.lookup_posting_offsets_one(document, *token_id)
            ).collect();
        if posting_offsets.len() == 0 { None } else { Some(posting_offsets) }
    }

    fn read_postings_one(&self, d: &Document, idx: usize, n: u32) -> Vec<Posting> {
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

    fn read_postings_many(
        &self, document: &Document, posting_offsets: &Vec<(usize, u32)>
    ) -> Vec<Posting> {
        assert!(posting_offsets.len() > 0, "Must contain offsets");
        if posting_offsets.len() == 1 {
            self.read_postings_one(document, posting_offsets[0].0, posting_offsets[0].1)
        } else {
            let m = &self.data[document.file_num];
            let time_int_size = self.time_int_size();
            let posting_size = self.posting_size();
            let base_ofs = document.base_offset + document.inv_index_offset;

            let read_single_posting = |idx| {
                let ofs = idx * posting_size + base_ofs;
                let time_int = self.read_time_int(m, ofs);
                let pos = self.read_datum(m, ofs + time_int_size) as usize;
                (ms_to_s(time_int.0), ms_to_s(time_int.1), pos, 1)
            };

            let mut iter_idxs = vec![0u32; posting_offsets.len()];
            let mut heap = BinaryHeap::new();
            let mut postings = Vec::with_capacity(
                posting_offsets.iter().map(|ofs| ofs.1).sum::<u32>() as usize);
            for i in 0..posting_offsets.len() {
                heap.push(HeapPosting {
                    origin: i,
                    posting: read_single_posting(posting_offsets[i].0)
                });
                iter_idxs[i] += 1;
            }

            while let Some(HeapPosting { posting, origin }) = heap.pop() {
                postings.push(posting);
                if iter_idxs[origin] < posting_offsets[origin].1 {
                    heap.push(HeapPosting {
                        origin: origin,
                        posting: read_single_posting(posting_offsets[origin].0)
                    });
                    iter_idxs[origin] += 1;
                }
            }
            postings
        }
    }

    fn check_contains_ngram(
        &self, ngram: &Vec<Token>, query_plan: &Vec<usize>, document: &Document
    ) -> bool {
        let ngram_len = ngram.len();
        assert!(ngram_len > 1, "Ngram must have > 1 tokens");

        let posting_offsets: Vec<Option<Vec<(usize, u32)>>> = ngram.iter().map(
            |token| self.lookup_posting_offsets_many(document, token)
        ).collect();

        // One of the tokens is not present in the document
        if posting_offsets.iter().any(|v| v.is_none()) {
            return false;
        }

        let init_pos = query_plan[0];
        let mut cand_idxs: Vec<usize> = self.read_postings_many(
            document, posting_offsets[init_pos].as_ref().unwrap()
        ).iter().filter_map(|p| {
            if p.2 < init_pos { None } else { Some(p.2 - init_pos) }
        }).collect();

        // Check candidate indices with postings at each position in the order of the query plan
        for i in 1..ngram_len {
            let pos = query_plan[i];
            let postings = self.read_postings_many(document, posting_offsets[pos].as_ref().unwrap());
            let postings_len = postings.len();
            let mut postings_iter_idx = 0;

            let mut new_cand_idxs = vec![];
            for cand_iter_idx in 0..cand_idxs.len() {
                let cand_idx = cand_idxs[cand_iter_idx];
                let exp_idx = cand_idx + pos;
                while postings_iter_idx < postings_len && postings[postings_iter_idx].2 < exp_idx {
                    postings_iter_idx += 1;
                }
                if postings_iter_idx == postings_len {
                    break;
                }

                let p2 = postings[postings_iter_idx];
                if p2.2 == exp_idx {
                    if i == ngram_len - 1 {
                        return true;    // Short circuit as soon as one is found
                    } else {
                        new_cand_idxs.push(cand_idx);
                    }
                }
            }
            if new_cand_idxs.len() == 0 {
                break;
            }
            cand_idxs = new_cand_idxs;
        }
        false
    }

    fn find_ngram_postings(
        &self, ngram: &Vec<Token>, query_plan: &Vec<usize>, document: &Document
    ) -> Option<Vec<Posting>> {
        let ngram_len = ngram.len();

        let posting_offsets: Vec<Option<Vec<(usize, u32)>>> = ngram.iter().map(
            |token| self.lookup_posting_offsets_many(document, token)
        ).collect();

        // One of the tokens is not present in the document
        if posting_offsets.iter().any(|v| v.is_none()) {
            return None;
        }

        let init_pos = query_plan[0];
        let mut postings1: Vec<Posting> = self.read_postings_many(
            document, posting_offsets[init_pos].as_ref().unwrap()
        ).iter().filter_map(|p| {
            if p.2 < init_pos { None } else { Some((p.0, p.1, p.2 - init_pos, ngram_len)) }
        }).collect();

        // Merge lists of postings in a pairwise manner according to the query plan
        for i in 1..ngram_len {
            let pos = query_plan[i];
            let postings2 = self.read_postings_many(
                document, posting_offsets[pos].as_ref().unwrap());
            let postings2_len = postings2.len();
            let mut posting2_iter_idx = 0;

            let mut new_postings = vec![];
            for postings1_iter_idx in 0..postings1.len() {
                let p1 = postings1[postings1_iter_idx];
                let exp_p2_idx = p1.2 + pos;
                while posting2_iter_idx < postings2_len && postings2[posting2_iter_idx].2 < exp_p2_idx {
                    posting2_iter_idx += 1;
                }
                if posting2_iter_idx == postings2_len {
                    break;
                }

                let p2 = postings2[posting2_iter_idx];
                if p2.2 == exp_p2_idx {
                    new_postings.push((
                        if p1.0 < p2.0 { p1.0 } else { p2.0 },  // min
                        if p1.1 < p2.1 { p2.1 } else { p1.1 },  // max
                        p1.2, ngram_len
                    ))
                }
            }
            if new_postings.len() == 0 {
                return None
            }
            postings1 = new_postings;
        }
        Some(postings1)
    }
}

#[pyclass]
pub struct RsCaptionIndex {
    _impl: _RsCaptionIndexImpl,
    debug: bool
}

#[pymethods]
impl RsCaptionIndex {

    fn document_exists(&self, doc_id: DocumentId) -> PyResult<bool> {
        Ok(self._impl.docs.get(&doc_id).is_some())
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
            match self._impl.lookup_posting_offsets_many(d, &unigram) {
                None => None,
                Some(pofs) => Some(self._impl.read_postings_many(d, &pofs))
            }
        };
        let docs_to_unigrams =
            if doc_ids.len() > 0 {
                doc_ids.par_sort();
                doc_ids.par_iter().filter_map(
                    |id| match self._impl.docs.get(&id) {
                        None => None,
                        Some(d) => match lookup_and_read_postings(d) {
                            None => None,
                            Some(p) => Some((*id, p))
                        }
                    }
                ).collect()
            } else {
                self._impl.docs.par_iter().filter_map(
                    |(id, d)| match lookup_and_read_postings(d) {
                        None => None,
                        Some(p) => Some((*id, p))
                    }
                ).collect()
            };
        Ok(docs_to_unigrams)
    }

    fn unigram_contains(
        &self, unigram: Token, mut doc_ids: Vec<DocumentId>
    ) -> PyResult<Vec<DocumentId>> {
        if self.debug {
            let len_str = doc_ids.len().to_string();
            eprintln!("unigram contains: [{:?}] in {} documents", unigram,
                      if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
        }
        let has_unigram = |d| unigram.iter().any(
            |t| self._impl.lookup_posting_offsets_one(d, *t).is_some());
        let docs_w_token =
            if doc_ids.len() > 0 {
                doc_ids.par_sort();
                doc_ids.par_iter().filter_map(
                    |id| match self._impl.docs.get(&id) {
                        None => None,
                        Some(d) => if has_unigram(d) {Some(*id)} else {None}
                    }
                ).collect()
            } else {
                self._impl.docs.par_iter().filter_map(
                    |(id, d)| if has_unigram(d) {Some(*id)} else {None}
                ).collect()
            };
        Ok(docs_w_token)
    }

    fn ngram_search(
        &self, ngram: Vec<Token>, mut doc_ids: Vec<DocumentId>, query_plan: Vec<usize>
    ) -> PyResult<Vec<(DocumentId, Vec<Posting>)>> {
        if ngram.len() <= 1 {
            Err(exceptions::ValueError::py_err("Ngram must have at least 2 tokens"))
        } else {
            if self.debug {
                let len_str = doc_ids.len().to_string();
                eprintln!("ngram search: {:?} in {} documents", ngram,
                          if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
            }
            let search_postings = |id, d| {
                match self._impl.find_ngram_postings(&ngram, &query_plan, d) {
                    None => None,
                    Some(p) => Some((id, p))
                }
            };
            let docs_to_ngrams =
                if doc_ids.len() > 0 {
                    doc_ids.par_sort();
                    doc_ids.par_iter().filter_map(
                        |id| match self._impl.docs.get(&id) {
                            None => None,
                            Some(d) => search_postings(*id, d)
                        }
                    ).collect()
                } else {
                    self._impl.docs.par_iter().filter_map(
                        |(id, d)| search_postings(*id, d)
                    ).collect()
                };
            Ok(docs_to_ngrams)
        }
    }

    fn ngram_contains(
        &self, ngram: Vec<Token>, mut doc_ids: Vec<DocumentId>, query_plan: Vec<usize>
    ) -> PyResult<Vec<DocumentId>> {
        if ngram.len() <= 1 {
            Err(exceptions::ValueError::py_err("Ngram must have at least 2 tokens"))
        } else {
            if self.debug {
                let len_str = doc_ids.len().to_string();
                eprintln!("ngram contains: {:?} in {} documents", ngram,
                          if doc_ids.len() > 0 {len_str.as_str()} else {"all"});
            }
            let has_ngram = |id, d| if self._impl.check_contains_ngram(&ngram, &query_plan, d) {
                Some(id)
            } else { None };
            let docs_w_ngram =
                if doc_ids.len() > 0 {
                    doc_ids.par_sort();
                    doc_ids.par_iter().filter_map(|id| match self._impl.docs.get(&id) {
                         None => None,
                         Some(d) => has_ngram(*id, d)
                    }).collect()
                } else {
                    self._impl.docs.par_iter().filter_map(|(id, d)| has_ngram(*id, d)).collect()
                };
            Ok(docs_w_ngram)
        }
    }

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
            _impl: _RsCaptionIndexImpl {
                docs: docs, data: index_mmaps, datum_size: datum_size,
                start_time_size: start_time_size, end_time_size: end_time_size
            },
            debug: debug
        });
        Ok(())
    }
}
