/* Memory mapped caption index */

use pyo3::prelude::*;
use pyo3::exceptions;
use std::cmp;
use std::mem;
use std::fs::File;
use memmap::{MmapOptions, Mmap};

use common::*;

struct _RsDocumentDataImpl {
    // The file containing the document index
    id: usize,

    // Length in milliseconds
    duration: Millis,

    // Offset of the time inteval index
    time_index_offset: usize,
    time_int_count: usize,

    // Offset of the raw tokens
    tokens_offset: usize,
    length: usize,

    // Encoding
    datum_size: usize,
    start_time_size: usize,
    end_time_size: usize,

    m: Mmap
}

impl _RsDocumentDataImpl {

    fn time_int_size(&self) -> usize {
        self.start_time_size + self.end_time_size
    }

    fn posting_size(&self) -> usize {
        self.time_int_size() + self.datum_size
    }

    fn read_datum(&self, i: usize) -> u32 {
        read_mmap(&self.m, i, self.datum_size)
    }

    fn read_time_int(&self, i: usize) -> (Millis, Millis) {
        let start = read_mmap(&self.m, i, self.start_time_size);
        let diff = read_mmap(&self.m, i + self.start_time_size, self.end_time_size);
        (start, start + diff)
    }

    fn lookup_time_int(&self, ms: Millis) -> Option<usize> {
        let mut min_idx: usize = 0;
        let mut max_idx = self.time_int_count as usize;
        let base_index_ofs = self.time_index_offset;
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
        if min_idx == (self.time_int_count as usize) {
            None
        } else {
            Some(min_idx)
        }
    }
}

#[pyclass]
pub struct RsDocumentData {
    _impl: _RsDocumentDataImpl,
    debug: bool
}

#[pymethods]
impl RsDocumentData {

    fn id(&self) -> PyResult<usize> {
        Ok(self._impl.id)
    }

    fn length(&self) -> PyResult<usize> {
        Ok(self._impl.length)
    }

    fn duration(&self) -> PyResult<f32> {
        Ok(ms_to_s(self._impl.duration))
    }

    fn tokens(&self, position: usize, n: usize) -> PyResult<Vec<TokenId>> {
        if self.debug {
            eprintln!("tokens: {}+{}", position, n);
        }
        let min_pos = cmp::min(position, self._impl.length);
        let max_pos = cmp::min(position + n, self._impl.length);
        let mut tokens = Vec::with_capacity(max_pos - min_pos);
        for pos in min_pos..max_pos {
            let ofs = pos * self._impl.datum_size + self._impl.tokens_offset;
            tokens.push(self._impl.read_datum(ofs));
        }
        Ok(tokens)
    }

    fn intervals(&self, start: Seconds, end: Seconds) ->  PyResult<Vec<Posting>> {
        if self.debug {
            eprintln!("intervals: {}s to {}s", start, end);
        }
        // Get document locations that overlap start and end
        if start > ms_to_s(u32::max_value()) {
            return Err(exceptions::ValueError::py_err("Start time exceeds maximum allowed"))
        }
        let start_ms = if start > 0. { s_to_ms(start) } else { 0 };
        let posting_size = self._impl.posting_size();
        let time_int_size = self._impl.time_int_size();

        let mut locations = vec![];
        match self._impl.lookup_time_int(start_ms) {
            Some(start_idx_immut) => {
                let mut start_idx = start_idx_immut;
                if start_idx > 0 {
                    start_idx -= 1;
                }
                let duration = self._impl.duration;
                let end_ms = if ms_to_s(duration) < end {duration} else {s_to_ms(end)};

                let time_int_count = self._impl.time_int_count;
                let base_index_ofs = self._impl.time_index_offset;
                let length = self._impl.length;
                for i in start_idx..(time_int_count as usize) {
                    let ofs = i * posting_size + base_index_ofs;
                    let time_int = self._impl.read_time_int(ofs);
                    if cmp::min(end_ms, time_int.1) >= cmp::max(start_ms, time_int.0) {
                        // Non-zero overlap
                        let pos = self._impl.read_datum(ofs + time_int_size) as usize;
                        let next_pos: usize = if i + 1 < (time_int_count as usize) {
                            self._impl.read_datum(ofs + posting_size + time_int_size) as usize
                        } else {length};
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
    }

    fn position(&self, time: Seconds) -> PyResult<Position> {
        if self.debug {
            eprintln!("position: {}s", time);
        }
        match self._impl.lookup_time_int(s_to_ms(time)) {
            Some(idx) => {
                let ofs = self._impl.time_index_offset +
                    idx * self._impl.posting_size() + self._impl.time_int_size();
                Ok(self._impl.read_datum(ofs) as Position)
            },
            None => Ok(self._impl.length as Position)
        }
    }

    #[new]
    unsafe fn __new__(obj: &PyRawObject, id: usize, data_path: String, datum_size: usize,
                      start_time_size: usize, end_time_size: usize, debug: bool
    ) -> PyResult<()> {
        let m: Mmap = MmapOptions::new().map(&File::open(&data_path).unwrap()).unwrap();

        let u32_size = mem::size_of::<u32>();
        let time_int_entry_size = datum_size + start_time_size + end_time_size;

        let doc_id = read_mmap_u32(&m, 0) as usize;
        if doc_id != id {
            return Err(exceptions::IOError::py_err("Document id does not match expected id"));
        }

        let duration: Millis = read_mmap_u32(&m, u32_size);
        let time_int_count = read_mmap_u32(&m, 2 * u32_size) as usize;
        let length = read_mmap_u32(&m, 3 * u32_size) as usize;
        let time_index_offset = 4 * u32_size;
        let tokens_offset = time_index_offset + time_int_count * time_int_entry_size;
        let total_len = tokens_offset + length * datum_size;

        if debug {
            eprintln!("Document: id={} duration={} intervals={} length={}",
                      doc_id, duration, time_int_count, length);
        }

        assert!(total_len == m.len(), "Incorrect byte offsets");

        obj.init(RsDocumentData {
            _impl: _RsDocumentDataImpl {
                id: doc_id, duration: duration,
                time_index_offset: time_index_offset, time_int_count: time_int_count,
                tokens_offset: tokens_offset, length: length,
                datum_size: datum_size, start_time_size: start_time_size,
                end_time_size: end_time_size, m: m
            },
            debug: debug
        });
        Ok(())
    }
}
