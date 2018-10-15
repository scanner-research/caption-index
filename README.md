# Caption Index

Simple library for building position, time, and inverted indexes on .srt files
for quick indexing and n-gram search. The goal of this module is provide
reasonably efficient iterators over caption/text files for analytics code in
Python, while limiting resident memory usage and IO/computational latency.

## Usage

#### Indexing your files

Run `scripts/build.py` and point it to the directory containing your subtitle
files to build an index. This can take some time and require significant
computation and memory resources.

#### Using your index

The `scripts` directory contains examples for how to use the various indices
that were built by `scripts/build.py`.

- `scripts/search.py` demonstrates n-gram and topic search in a command line
  application.

- `scripts/scan.py` performs a scan over all of the tokens in all documents.

- `scripts/build_metadata.py` produces part-of-speech metadata and stores it
  in an indexable format.

## Tests

Run `test/tests.py`.
