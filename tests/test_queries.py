"""
Test text query parsing.
"""

import yaml
import captions.query as query


def test_query_parser():
    queries = [
        'hello world testing',
        'hello [ world good morning ] america',
        '(hello  world)',
        '(hello \tworld)',
        '(hello world)',
        'hello & world',
        'hello & world :: 15',
        'hello & world // 15',
        'hello | world',
        'hello \\ world',
        'hello \\ world :: 123',
        'hello \\ world // 123',
        'the & (red lobster | (soaring & bald eagle))',
        '[the] red [kittens]',
        '(the & (red | blue) & (cat \\ sat on)) | a [green mat]',
        '(the & (red | blue) & (cat \\ sat on :: 24) :: 12) | a [green mat]',
        '(the & (red | blue) & (cat \\ sat on // 24) // 12) | a [green mat]',
        'U.S | U.K',
        'red-black tree'
    ]

    for raw_query in queries:
        print('Raw query:', raw_query)
        q = query.Query(raw_query)
        print(yaml.dump(q._tree._pprint_data, indent=4))
