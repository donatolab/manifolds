from setuptools import setup

setup(
    name="donlabtools",
    version="0.22",
    description="donato lab [ca] imaging tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/donatolab/manifolds",
    author="https://github.com/donatolab/manifolds",
    author_email="<mitelutco@gmail.com>",
    packages=["donlabtools"],
    install_requires=[
        'numpy',
        "matplotlib",
        'parmap',
        'tqdm',
        'scipy',
        'opencv-python',
        'scikit-learn',
        'networkx',
        'pandas'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

