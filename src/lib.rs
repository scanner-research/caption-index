extern crate pyo3;
extern crate memmap;
extern crate byteorder;
extern crate indicatif;
extern crate subparse;
extern crate rayon;

use pyo3::prelude::*;
use pyo3::Python;
use pyo3::{wrap_pyfunction,wrap_pymodule};
use std::collections::HashMap;

mod common;
mod index;
mod indexer;
mod data;

use index::RsCaptionIndex;
use data::RsDocumentData;

#[pyfunction]
fn tokenize(s: String) -> Vec<String> {
    indexer::tokenize(&s)
}

#[pyfunction]
fn count_tokens(doc_paths: Vec<String>, max_token_len: usize, batch_size: usize, is_aligned: bool) -> HashMap<String, usize> {
    indexer::count_tokens(&doc_paths, max_token_len, batch_size, is_aligned)
}

#[pyfunction]
fn index_documents(
    index_and_doc_paths: Vec<(String, Vec<(usize, String, String)>)>, lexicon: HashMap<String, u32>,
    is_aligned: bool, datum_size: usize, start_time_size: usize, end_time_size: usize
) -> () {
    indexer::index_documents(&index_and_doc_paths, &lexicon, is_aligned,
                             datum_size, start_time_size, end_time_size)
}

#[pyfunction]
fn set_parallelism(n: usize) -> () {
    indexer::set_parallelism(n);
}

#[pymodule]
fn indexer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(set_parallelism))?;
    m.add_wrapped(wrap_pyfunction!(count_tokens))?;
    m.add_wrapped(wrap_pyfunction!(index_documents))?;
    Ok(())
}

#[pymodule]
fn rs_captions(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RsCaptionIndex>()?;
    m.add_class::<RsDocumentData>()?;
    m.add_wrapped(wrap_pyfunction!(tokenize))?;
    m.add_wrapped(wrap_pymodule!(indexer))?;
    Ok(())
}
