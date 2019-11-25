# Caption Index

Simple library for building position, time, and inverted indexes on .srt files
for quick indexing and n-gram search. The goal of this module is provide
reasonably efficient iterators over caption/text files for analytics code,
while limiting resident memory usage and IO/computational latency.

## Install

First, install Rust. Configure Rust nightly in the caption-index
code directory (tested with version 1.39.0). On Linux, this can be set with
`rustup override set nightly-2019-09-01` in the cloned repository directory.

Run `python3 setup.py install --user`.

## Usage

#### Indexing your files

Run `scripts/build_index.py` and point it to the directory containing your
subtitle files to build an index. This can take some time and require
significant computation and memory resources if there are many files
(i.e., hundreds of thousands).

After the indexer has run, there will be three files in the index directory.
These are:
  - `documents.txt`
  - `lexicon.txt`
  - `index.bin`

Note that if you set ran the indexer with the `--chunk-size` set, then
`index.bin` will be a directory containing the index files.

#### Updating an index

Sometimes, we may need to index additional documents after we first built our
index. To do this, run `scripts/update_index.py`. This will use the same
lexicon as the original index (to update the lexicon, you must rebuild the
index from scratch).

#### Using your index

The `tools` directory contains examples for how to use the various indices
that were built by `scripts/build_index.py`.

- `tools/search.py` demonstrates n-gram and topic search in a command line
  application.

- `tools/scan.py` performs a scan over all of the tokens in all documents.

- `tools/index_metadata.py` produces part-of-speech metadata and stores it
  in a searchable format.

- `tools/index_ngram_counts.py` produces n-gram frequencies across the entire
  data set.

- `tools/compute_pmi_lexicon.py` builds topic lexicons around n-grams using a
  PMI topic model.

## Tests

Run `pytest -v tests`
