from setuptools import setup

setup_requires = ['pytest-runner']
install_requires = [
    'msgpack==0.5.6',
    'numpy>=1.15.4',
    'pysrt>=1.1.1',
    'pytest>=4.0.1',
    'spacy>=2.0.18',
    'tqdm>=4.28.1'
]
tests_require = install_requires + ["pytest"]

setup(
    name="caption-index",
    version="0.1.0",
    classifiers=[],
    packages=["captions"],
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False
)