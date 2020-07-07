/* Indexer utils in Rust */

use std::collections::{HashMap,BTreeMap};
use std::fs::File;
use std::io::prelude::*;
use std::path::{PathBuf, Path};
use std::cmp;
use byteorder::{ByteOrder, LittleEndian};
use subparse::{get_subtitle_format, parse_str};
use indicatif::ProgressBar;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use common::*;


static SPLIT_DELIMITERS: &'static [char] = &[
    '.', ',', '!', '?', ':', ';', '(', ')',
    '{', '}', '[', ']', '`', '|', '"', '\''
];

pub fn set_parallelism(n: usize) -> () {
    ThreadPoolBuilder::new().num_threads(n).build_global().unwrap();
}

fn read_file(path: &Path) -> Option<String> {
    let mut file = File::open(path).unwrap();
    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Ok(_) => Some(s),
        Err(e) => {
            println!("Failed to read: {} - {}", path.display(), e);
            None
        }
    }
}

pub fn tokenize(s: &String) -> Vec<String> {
    let mut result = Vec::new();
    let mut last = 0;
    for (index, matched) in s.match_indices(
        |c: char| c.is_whitespace() || SPLIT_DELIMITERS.contains(&c)
    ) {
        if last != index {
            result.push(s[last..index].to_string());
        }
        let matched_str = matched.to_string();
        if !matched_str.trim().is_empty() {
            result.push(matched_str);
        }
        last = index + matched.len();
    }
    if last < s.len() {
        result.push(s[last..].to_string());
    }
    result
}

#[inline]
fn has_misalignment_indicators(s: &String) -> bool {
    let l = s.len();
    l >= 2 && s.starts_with('{') && s.ends_with('}')
}

#[inline]
fn line_to_tokens(s: &String, is_aligned: bool) -> Vec<String> {
    if is_aligned && has_misalignment_indicators(&s) {
        tokenize(&s[1..s.len() - 1].to_string())
    } else {
        tokenize(&s)
    }
}

fn count_tokens_part(doc_paths: &Vec<String>, max_token_len: usize, is_aligned: bool) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for doc_path in doc_paths {
        let path = PathBuf::from(doc_path);
        match read_file(&path) {
            Some(file_content) => {
                let format = get_subtitle_format(path.extension(), file_content.as_bytes()).expect("unknown format");
                let subtitle_file = parse_str(format, &file_content, 0.).expect("parser error");
                let subtitle_entries = subtitle_file.get_subtitle_entries().expect("unexpected error");
                for token in subtitle_entries.iter().filter_map(|x| x.line.as_ref()).flat_map(
                    |x| line_to_tokens(&x, is_aligned)
                ).filter(
                    |t| t.len() <= max_token_len
                ) {
                    let count = counts.entry(token).or_insert(0usize);
                    *count += 1;
                }
            },
            None => {}
        }
    }
    counts
}

pub fn count_tokens(doc_paths: &Vec<String>, max_token_len: usize, batch_size: usize,
                    is_aligned: bool) -> HashMap<String, usize> {
    let mut doc_batches = Vec::new();
    for i in 0..(doc_paths.len() as f64 / batch_size as f64).ceil() as usize {
        let base_idx = i * batch_size;
        doc_batches.push(doc_paths[base_idx..cmp::min(base_idx + batch_size, doc_paths.len())].to_vec());
    }

    let pbar = ProgressBar::new(doc_paths.len() as u64);
    pbar.tick();

    let mut all_counts = HashMap::new();
    for part_counts in doc_batches.par_iter().map(|d| {
        let result = count_tokens_part(&d, max_token_len, is_aligned);
        pbar.inc(d.len() as u64);
        result
    }).collect::<Vec<HashMap<String, usize>>>().iter() {
        for (token, n) in part_counts {
            let count = all_counts.entry(token.clone()).or_insert(0usize);
            *count += n;
        }
    }
    all_counts
}

#[inline]
fn write_u32(f: &mut File, v: u32) -> () {
    let mut buf = vec![0u8; 4];
    LittleEndian::write_u32(&mut buf, v);
    f.write_all(&buf).unwrap();
}

#[inline]
fn write_data(f: &mut File, mut v: u32, datum_size: usize) -> () {
    let mut buf = vec![0u8; datum_size];
    for i in 0..datum_size {
        buf[i] = v as u8;
        v = v >> 8;
    }
    assert!(v == 0);
    f.write_all(&buf).unwrap();
}

#[inline]
fn write_time_interval(
    f: &mut File, start: u32, end: u32, start_time_size: usize, end_time_size: usize
) -> () {
    let delta = end - start;
    write_data(f, start, start_time_size);
    write_data(f, delta, end_time_size);
}

fn write_inverted_index(
    f: &mut File, doc_id: usize, inverted_idx: &BTreeMap<TokenId, Vec<(Position, Millis, Millis)>>,
    num_postings: usize, datum_size: usize, start_time_size: usize, end_time_size: usize
) -> () {
    write_u32(f, doc_id as u32);
    write_u32(f, inverted_idx.len() as u32);
    write_u32(f, num_postings as u32);
    let mut i = 0;
    for (token_id, count) in inverted_idx.iter().map(|(a, b)| (a, b.len())) {
        write_data(f, *token_id, datum_size);
        write_data(f, i as u32, datum_size);
        i += count;
    }
    assert!(i == num_postings);
    for (_, postings) in inverted_idx {
        for (position, start, end) in postings {
            write_time_interval(f, *start, *end, start_time_size, end_time_size);
            write_data(f, *position as u32, datum_size);
        }
    }
}

fn write_binary_data(
    out_path: &String, doc_id: usize, lines: &Vec<(Position, Millis, Millis, Vec<TokenId>)>,
    duration: u32, num_tokens: usize,
    datum_size: usize, start_time_size: usize, end_time_size: usize
) -> () {
    let mut f = File::create(out_path).expect("error writing file");
    write_u32(&mut f, doc_id as u32);
    write_u32(&mut f, duration as u32);
    write_u32(&mut f, lines.len() as u32);
    write_u32(&mut f, num_tokens as u32);
    for (position, start, end, _) in lines {
        write_time_interval(&mut f, *start, *end, start_time_size, end_time_size);
        write_data(&mut f, *position as u32, datum_size);
    }
    let mut i = 0;
    for token in lines.iter().flat_map(|x| x.3.iter()) {
        write_data(&mut f, *token, datum_size);
        i += 1;
    }
    assert!(i == num_tokens);
}

pub fn index_documents(
    index_and_doc_paths: &Vec<(String, Vec<(usize, String, String)>)>,
    lexicon: &HashMap<String, u32>, is_aligned: bool,
    datum_size: usize, start_time_size: usize, end_time_size: usize
) -> () {
    let max_datum_value = 2u32.pow(datum_size as u32 * 8) - 1;
    let max_time_interval = 2u32.pow(end_time_size as u32 * 8) - 1;

    let pbar = ProgressBar::new(index_and_doc_paths.iter().map(|x| x.1.len() as u64).sum());
    pbar.tick();

    index_and_doc_paths.par_iter().for_each(|(index_path, docs)| {
        let mut f = File::create(index_path).expect("Unable to open file");

        let mut neg_interval_count = 0;
        let mut long_interval_count = 0;

        for (doc_id, doc_path, data_path) in docs {
            pbar.inc(1);
            let path = PathBuf::from(doc_path);
            match read_file(&path) {
                Some(file_content) => {
                    let format = get_subtitle_format(path.extension(), file_content.as_bytes()).expect("unknown format");
                    let subtitle_file = parse_str(format, &file_content, 0.).expect("parser error");
                    let subtitle_entries = subtitle_file.get_subtitle_entries().expect("unexpected error");

                    let mut num_tokens = 0usize;
                    let mut doc_lines: Vec<(Position, Millis, Millis, Vec<TokenId>)> = Vec::new();
                    let mut doc_inv_index: BTreeMap<TokenId, Vec<(Position, Millis, Millis)>> = BTreeMap::new();
                    let mut doc_num_postings = 0usize;
                    let mut doc_duration = 0u32;

                    for subtitle_entry in subtitle_entries.iter().filter(|x| x.line.is_some()) {
                        let start = subtitle_entry.timespan.start.msecs() as Millis;
                        let mut end = subtitle_entry.timespan.end.msecs() as Millis;
                        if start > end {
                            if neg_interval_count == 0 {
                                println!("Warning: start time > end time ({} > {})", start, end);
                            }
                            end = start;
                            neg_interval_count += 1;
                        }
                        if end - start > max_time_interval {
                            if long_interval_count == 0 {
                                println!("Warning: end - start > {}ms", max_time_interval);
                            }
                            end = start + max_time_interval;
                            long_interval_count += 1;
                        }
                        let line = subtitle_entry.line.as_ref().unwrap();
                        let token_ids: Vec<TokenId> = line_to_tokens(&line, is_aligned).iter().map(
                            |t| match lexicon.get(t) {
                                Some(id) => *id as TokenId,
                                None => max_datum_value
                            }
                        ).collect();
                        let token_count = token_ids.len();
                        for (j, token_id) in token_ids.iter().enumerate() {
                            if *token_id != max_datum_value {
                                let postings = doc_inv_index.entry(*token_id).or_insert(vec![]);
                                postings.push(((num_tokens + j) as Position, start, end));
                                doc_num_postings += 1;
                            }
                        }
                        doc_lines.push((num_tokens as Position, start, end, token_ids));
                        num_tokens += token_count;
                        doc_duration = cmp::max(end, doc_duration);
                    }
                    write_inverted_index(&mut f, *doc_id, &doc_inv_index, doc_num_postings,
                                         datum_size, start_time_size, end_time_size);
                    write_binary_data(data_path, *doc_id, &doc_lines, doc_duration, num_tokens,
                                      datum_size, start_time_size, end_time_size);
                },
                None => ()
            }
        }

        if long_interval_count + neg_interval_count > 0 {
            println!("Warning: supressed error messages for {} negative and {} long intervals",
                     neg_interval_count, long_interval_count);
        }
    });
}
