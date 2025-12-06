from setuptools import find_packages, setup

__version__ = "0.1"

setup(
    name="virtual_streamer",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=[],
    version=__version__,
    description="An application package for streaming video avatars",
    author="admor",
)
