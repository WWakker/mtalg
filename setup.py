import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

about = {}
with open("mtalg/__about__.py") as f:
    exec(f.read(), about)

with open("mtalg/requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="mtalg",
    version=about['__version__'],
    author=about['__authors__'],
    author_email=about['__email__'],
    description=about['__about__'],
    url=about['__url__'],
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require={'full': ['numba']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
