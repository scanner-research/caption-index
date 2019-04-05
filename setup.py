import sys

from setuptools import setup

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, '-m', 'pip', 'install', 'setuptools-rust'])
    if errno:
        print('Please install setuptools-rust package')
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension

setup_requires = ['setuptools-rust', 'wheel', 'pytest-runner']
install_requires = [
    'numpy>=1.15.4',
    'parsimonious>=0.8.1',
    'pysrt>=1.1.1',
    'pytest>=4.0.1',
    'PyYAML>=3.13',
    'spacy>=2.0.18',
    'termcolor>=1.1.0',
    'tqdm>=4.28.1',
]
tests_require = install_requires + ['pytest>=4.2.1', 'pyaml>=18.11.0']

setup(
    name='caption-index',
    version='0.2.0',
    classifiers=[],
    packages=['captions'],
    rust_extensions=[RustExtension('captions.rs_captions', 'Cargo.toml')],
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False
)
