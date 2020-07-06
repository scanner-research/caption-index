# Caption Index

Simple library for building position, time, and inverted indexes on .srt files
for quick indexing and n-gram search. The goal of this module is provide
reasonably efficient iterators over caption/text files for analytics code,
while limiting resident memory usage and IO/computational latency.

## Install

First, install Rust (tested on stable 1.43.0). Run `python3 setup.py install
--user`.

## Usage

#### Indexing your files

Run `scripts/build_index.py` and point it to the directory containing your
subtitle files to build an index. This can take some time and require
significant computation and memory resources if there are many files
(i.e., hundreds of thousands).

After the indexer has run, there will be four entries in the index directory.
These are:
  - `documents.txt`
  - `lexicon.txt`
  - `index.bin`
  - `data/`

Note that if you set ran the indexer with the `--chunk-size` set, then
`index.bin` will be a directory containing the index files.

`data` is a directory containing binary encoded captions, one per file, and
named by the document id. Do not manually rename these files!

#### Updating an index

Sometimes, we may need to index additional documents after we first built our
index. To do this, run `scripts/update_index.py`. You can optionally also
update the lexicon.

#### Using your index

The `tools` directory contains examples for how to use the various indices
that were built by `scripts/build_index.py`.

- `tools/search.py` demonstrates n-gram and topic search in a command line
  application.

- `tools/scan.py` performs a scan over all of the tokens in all documents.

## Tests

Run `pytest -v` from inside the `tests` directory.

If it complains that spaCy's "en" is missing, you likely need to run
`python3 -m spacy download en`.
