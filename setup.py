import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visual_clutter",
    version="1.0.6",
    author="Amir Hossein Kargaran",
    author_email="kargaranamir@gmail.com",
    description="Python implementation of two measures of visual clutter (Feature Congestion and Subband Entropy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kargaranamir/visual-clutter",
    project_urls={
        "Bug Tracker": "https://github.com/kargaranamir/visual-clutter/issues",
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ],
    install_requires=required,
    packages=setuptools.find_packages(),
    keywords = ['visual clutter', 'feature congestion', 'subband entropy'],
    python_requires=">=3.6",
)
