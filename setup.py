import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visual_clutter",
    version="1.0.2",
    author="Amir Hossein Kargaran",
    author_email="kargaranamir@gmail.com",
    description="Python implementation of two measures of visual clutter (Feature Congestion and Subband Entropy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kargaranamir/visual_clutter",
    project_urls={
        "Bug Tracker": "https://github.com/kargaranamir/visual_clutter/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux or OSX",
    ],
    install_requires=[
        'numpy >= 1.20.2',
        'opencv-python >= 4.5.3',
	'scipy == 1.7.0',
	'Pillow >= 8.3.1',
	'pyrtools == 1.0.0',
	'scikit-image >= 0.16.2'
    ],
    packages=setuptools.find_packages(),
    keywords = ['visual clutter', 'feature congestion', 'subband entropy'],
    python_requires=">=3.6",
)
