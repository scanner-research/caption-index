#!/usr/bin/env python3

import argparse
import os
import pysrt
from tqdm import tqdm
from threading import Lock
from multiprocessing import Pool
from collections import defaultdict

from util.index import DocumentIndex, tokenize, UNKNOWN_TOKEN_ID


N_WORKERS = None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('subdir', type=str)
    p.add_argument('-o', dest='outpath', type=str, default='index.bin')
    p.add_argument('-j', dest='workers', type=int, default=8)
    p.add_argument('--debug', dest='debug', action='store_true')
    return p.parse_args()


def list_subs(dir):
    return [f for f in os.listdir(dir) if f.endswith('.srt')]


def load_srt(docpath):
    try:
        subs = pysrt.open(docpath)
    except:
        try:
            subs = pysrt.open(docpath, encoding='iso-8859-1')
        except:
            raise Exception(f'Cannot parse {docpath}')
    return subs


def get_doc_words(docpath):
    try:
        subs = load_srt(docpath)
    except Exception as e:
        print(e)
        return set()

    words = set()
    for s in subs:
        tokens = tokenize(s.text)
        words.update(tokens)
    return words


def get_words(subdir, subfiles):
    words = set()
    wordsLock = Lock()
    with tqdm(total=len(subfiles)) as pbar:
        pbar.set_description('Building lexicon')

        def collect(result):
            with wordsLock:
                words.update(result)
            pbar.update(1)

        pool = Pool(processes=N_WORKERS)

        for subfile in subfiles:
            docpath = os.path.join(subdir, subfile)
            pool.apply_async(get_doc_words, (docpath,), callback=collect)

        pool.close()
        pool.join()

    print(f'Lexicon size: {len(words)}')
    return words


def get_doc_index(sub_id, docpath, rlexicon):
    """
    Returns: {
        token : [
            (position, startTime, endTime), ...
        ]
    }
    """
    try:
        subs = load_srt(docpath)
    except Exception as e:
        print(e)
        return {}
    doc_index = defaultdict(list)
    position = 0
    for s in subs:
        start, end = s.start.ordinal / 1000., s.end.ordinal / 1000.
        for t in tokenize(s.text):
            tid = rlexicon.get(t, UNKNOWN_TOKEN_ID)
            doc_index[tid].append((position, start, end))
            position += 1
    return sub_id, doc_index


def build_index(subdir, subfiles, lexicon):
    # {
    #     token: { doc_id: "list of occurrences" },
    # }
    rlexicon = { w : i for i, w in enumerate(lexicon) }
    indexLock = Lock()
    index = {}
    with tqdm(total=len(subfiles)) as pbar:
        pbar.set_description('Building index')

        def collect(result):
            with indexLock:
                sub_id, doc_index = result
                for t, l in doc_index.items():
                    if t not in index:
                        index[t] = {}
                    index[t][sub_id] = l
            pbar.update(1)

        pool = Pool(processes=N_WORKERS)

        for i, subfile in enumerate(subfiles):
            docpath = os.path.join(subdir, subfile)
            pool.apply_async(get_doc_index, (i, docpath, rlexicon), callback=collect)

        pool.close()
        pool.join()
    return index


def main(subdir, outpath, workers, debug):
    global N_WORKERS
    N_WORKERS = workers

    subfiles = list(sorted(list_subs(subdir)))
    if debug:
        subfiles = subfiles[:100]

    lexicon = list(sorted(get_words(subdir, subfiles)))
    index = build_index(subdir, subfiles, lexicon)
    DocumentIndex(index=index, doclist=subfiles, lexicon=lexicon).store(outpath)


if __name__ == '__main__':
    main(**vars(get_args()))
