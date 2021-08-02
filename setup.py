import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visual_clutter",
    version="1.0.0",
    author="Amir Hossein Kargaran",
    author_email="kargaranamir@gmail.com",
    description="Python Implementation of two measures of visual clutter (Feature Congestion and Subband Entropy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kargaranamir/visual_clutter",
    project_urls={
        "Bug Tracker": "https://github.com/kargaranamir/visual_clutter/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux/Mac Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
