from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name="question_score",
    version="0.0.3",
    description="library for question evaluation including KDA, Knowledge Dependent Answerability introduced in EMNLP 2022 work.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Riiid NLP team",
    author_email="ai_nlp@riiid.co",
    url="https://github.com/riiid/question-score",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0."
            " https://creativecommons.org/licenses/by-nc-sa/4.0/",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
