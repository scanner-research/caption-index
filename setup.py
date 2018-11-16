from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = []

    def run(self):
        self.run_command("test_rust")

        import subprocess

        subprocess.check_call(["pytest", "tests"])


setup_requires = []
install_requires = ['msgpack', 'numpy', 'pysrt', 'spacy', 'tqdm']
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
    zip_safe=False,
    cmdclass=dict(test=PyTest),
)
