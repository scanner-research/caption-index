/* Memory mapped metadata file */

use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::PyBytes;
use pyo3::Python;
use std::collections::BTreeMap;
use std::cmp;
use std::mem;
use std::fs::File;
use memmap::{MmapOptions, Mmap};

use common::*;

#[pyclass]
pub struct RsMetadataIndex {
    docs: BTreeMap<DocumentId, (usize, usize)>,  // Offset and length
    data: Mmap,
    entry_size: usize,
    debug: bool
}

#[pymethods]
impl RsMetadataIndex {

    fn metadata(
        &self, doc_id: DocumentId, position: usize, n: usize
    ) -> PyResult<Vec<Py<PyBytes>>> {
        if self.debug {
            eprintln!("Metdata: {}+{} in {}", position, position, doc_id);
        }
        match self.docs.get(&doc_id) {
            Some((doc_ofs, doc_len)) => {
                let mut result = vec![];
                let max_idx = cmp::min(position + n, *doc_len);
                let data = self.data[*doc_ofs..*doc_ofs + max_idx * self.entry_size].as_ref();
                let gil = Python::acquire_gil();
                let py = gil.python();
                for i in position..max_idx {
                    let ofs = i * self.entry_size;

                    // This feels very unsafe...
                    let bytes = PyBytes::new(py, &data[ofs..ofs + self.entry_size]);
                    result.push(unsafe { Py::from_owned_ptr(bytes.into_ptr()) });
                }
                Ok(result)
            },
            None => Err(exceptions::ValueError::py_err("Document not found"))
        }
    }

    #[new]
    unsafe fn __new__(
        obj: &PyRawObject, meta_file: String, entry_size: usize, debug: bool
    ) -> PyResult<()> {
        let parse_meta = |m: &Mmap| {
            let mut docs = BTreeMap::new();
            let meta_size = m.len();
            let mut curr_offset = 0;
            let u32_size = mem::size_of::<u32>();
            while curr_offset < meta_size {
                let doc_id = read_mmap_u32(&m, curr_offset) as DocumentId;
                let n = read_mmap_u32(&m, curr_offset + u32_size) as usize;
                docs.insert(doc_id, (curr_offset, n));
                curr_offset += 2 * u32_size + n * entry_size;
            }
            if debug {
                eprintln!("Loaded index containing {} documents", docs.len());
            }
            assert!(curr_offset == meta_size, "Invalid number of bytes read");
            docs
        };

        let mmap = MmapOptions::new().map(&File::open(&meta_file)?);
        match mmap {
            Ok(m) => {
                obj.init(RsMetadataIndex {
                    docs: parse_meta(&m), data: m, entry_size: entry_size, debug: debug
                });
                Ok(())
            },
            Err(s) => Err(exceptions::Exception::py_err(s.to_string()))
        }
    }
}
