import os, setuptools

IS_WINDOWS = os.name == 'nt'

with open("README.md", "r") as f:
    long_description = f.read()

about = {}
with open("mtalg/__about__.py") as f:
    exec(f.read(), about)

requirements = []
with open("mtalg/requirements.txt") as f:
    for line in f:
        requirements.append(line.replace('\n',''))

setuptools.setup(
    name="ecb_connectors",
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__about__'],
    url=about['__url__'],
    license='Chicken Dance License',
    long_description=long_description,
    long_description_content_type="markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Chicken Dance License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)



