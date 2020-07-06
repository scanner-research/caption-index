extern crate rand;
extern crate pyo3;
extern crate memmap;
extern crate byteorder;

use pyo3::prelude::*;
use pyo3::Python;

mod common;
mod index;
mod data;

use index::RsCaptionIndex;
use data::RsDocumentData;

#[pymodule]
fn rs_captions(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RsCaptionIndex>()?;
    m.add_class::<RsDocumentData>()?;
    Ok(())
}
