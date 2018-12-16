import sys

from setuptools import setup

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, "-m", "pip", "install", "setuptools-rust"])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension


setup_requires = ["setuptools-rust", "wheel", "pytest-runner"]
install_requires = ['msgpack', 'numpy', 'pysrt', 'spacy', 'tqdm']
tests_require = install_requires + ["pytest"]

setup(
    name="caption-index",
    version="0.2.0",
    classifiers=[],
    packages=["captions"],
    rust_extensions=[RustExtension("captions.rs_captions", "Cargo.toml")],
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False
)
