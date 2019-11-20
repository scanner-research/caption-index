#![feature(specialization)]

extern crate rayon;
extern crate rand;
extern crate pyo3;
extern crate memmap;
extern crate byteorder;

use pyo3::prelude::*;
use pyo3::Python;

mod common;
mod index;
mod metadata;

use index::RsCaptionIndex;
use metadata::RsMetadataIndex;

#[pymodule]
fn rs_captions(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RsCaptionIndex>()?;
    m.add_class::<RsMetadataIndex>()?;
    Ok(())
}
