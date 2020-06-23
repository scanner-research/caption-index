
use std::mem;
use std::io::Cursor;
use byteorder::{ReadBytesExt, LittleEndian};
use memmap::Mmap;

pub type DocumentId = u32;
pub type TokenId = u32;
pub type Token = Vec<TokenId>;
pub type Seconds = f32;
pub type Millis = u32;
pub type Position = u32;


#[inline]
pub fn ms_to_s(ms: Millis) -> Seconds {
    (ms as f32) / 1000.
}

#[inline]
pub fn s_to_ms(s: Seconds) -> Millis {
    (s * 1000.) as u32
}

#[inline]
pub fn read_mmap_u32(m: &Mmap, i: usize) -> u32 {
    let mut rdr = Cursor::new(&m[i..i + mem::size_of::<u32>()]);
    rdr.read_u32::<LittleEndian>().unwrap()
}

#[inline]
pub fn read_mmap(m: &Mmap, i: usize, n: usize) -> u32 {
    assert!(n <= mem::size_of::<u32>(), "Cannot read more than u32");
    let mut result = 0;
    for j in 0..n {
        result += (m[i + j] as u32) << (j * 8);
    }
    result
}
